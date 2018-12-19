from __future__ import print_function
from __future__ import division

import sys
import time
from utils.data_loader import DataLoader
import numpy as np
from copy import deepcopy

import tensorflow as tf
from utils.attention_gru_cell import AttentionGRUCell
from model.encoder import Encoder
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops


def _add_gradient_noise(t, stddev=1e-3, name=None):
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)


def _position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le - 1)/2) * (j - (ls-1)/2)
    encoding = 1+ 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


class DMN_PLUS(object):
    def __init__(self, embedding_input, input_mask, embedding_question, vocab_size, max_mask_length, params):
        self.embedding_input = embedding_input
        self.input_mask = input_mask
        self.embedding_question = embedding_question
        self.vocab_size = vocab_size
        self.params = params
        self.encoder = Encoder(encoder_type=params.model.encoder_type,
                          num_layers=params.model.num_layers,
                          cell_type=params.model.cell_type,
                          num_units=params.model.num_units,
                          dropout=params.model.dropout)
        self.max_mask_length= max_mask_length
        self.output = self.inference()

    def get_predictions(self, output):
       preds = tf.nn.softmax(output)
       pred = tf.argmax(preds, 1)
       return pred

    def add_loss_op(self, output, labels):
       loss = tf.reduce_sum(
           tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=labels))
       for v in tf.trainable_variables():
           if not 'bias' in v.name.lower():
               loss += self.params.train.learning_rate
       tf.summary.scalar('loss', loss)
       return loss

    def add_training_op(self, loss):
        opt = tf.train.AdamOptimizer(learning_rate=self.params.train.learning_rate)
        gvs = opt.compute_gradients(loss)

        if self.params.model.cap_grads:
            gvs = [(tf.clip_by_norm(grad, self.params.model.max_grad_val), var) for grad, var in gvs]
        if self.params.model.noisy_grads:
            gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]
        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_representation(self):
        question_length = tf.reduce_sum(tf.to_int32(
            tf.not_equal(tf.reduce_max(self.embedding_question, axis=2), self.params.data.PAD_ID)), axis=1)
        _, question = self.encoder.build(self.embedding_question, question_length,scope="encoder")
        return question[0]

    def get_input_representation(self):
        self.input_length = tf.reduce_max(self.input_mask, axis=1)
        input_encoder_outputs, _ = self.encoder.build(self.embedding_input, self.input_length, encoder_type="UNI", scope="encoder")

        with tf.variable_scope("facts") as scope:
            print(self.input_mask.shape)
            batch_size = tf.shape(self.input_mask)[0]
            max_mask_length = tf.shape(self.input_mask)[1]
            input_mask = self.input_mask


            def get_encoded_fact(i):
                # print(i)
                mask_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(input_mask[i],
                                                                      self.params.data.PAD_ID)), axis=0)
                input_mask_temp = tf.boolean_mask(input_mask[i], tf.sequence_mask(mask_lengths, max_mask_length))

                encoded_facts = tf.gather_nd(input_encoder_outputs[i], tf.reshape(input_mask_temp, [-1, 1]))
                padding = tf.zeros(tf.stack([max_mask_length - mask_lengths, self.params.model.num_units]))
                return tf.concat([encoded_facts, padding], 0)

            facts_stacked = tf.map_fn(get_encoded_fact, tf.range(start=0, limit=batch_size), dtype=tf.float32)

            facts = tf.unstack(tf.transpose(facts_stacked, [1, 0, 2]), num=self.max_mask_length)
        return facts

    def build_input_module(self):
        facts = self.get_input_representation()
        question = self.get_question_representation()
        return facts, question

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self.params.model.embed_dim,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")

        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""

        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(fact_vecs)]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        tmp = tf.transpose(tf.stack(fact_vecs), [1, 0, 2])
        gru_inputs = tf.concat([tmp, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.params.model.num_units),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_length
                                           )

        return episode

    def add_answer_module(self, rnn_output, q_vec):
        """Linear softmax answer module"""
        if self.params.model.dropout:
            rnn_output = tf.nn.dropout(rnn_output, self.params.model.dropout)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                 self.vocab_size,
                                 activation=None)

        return output

    def inference(self):
        """Performs inference on the DMN model"""

        # input fusion module
        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
            print('==> get input representation')
            fact_vecs = self.get_input_representation()

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
            print('==> get question representation')
            q_vec = self.get_question_representation()


        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
            print('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for i in range(self.params.model.num_hops):
                # get a new episode
                print('==> generating episode', i)
                episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.params.model.num_units,
                                                  activation=tf.nn.relu)

            output = prev_memory

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE):
            output = self.add_answer_module(output, q_vec)

        return output


# if __name__ == '__main__':
#     from hbconfig import Config
#     import os
#
#     config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config')
#     params_path = os.path.join(config_path, 'bAbi_task1.yml')
#     Config(params_path)
#     print("Config: ", Config)
#     print(params_path)
#     data_loader = DataLoader(
#         task_path=Config.data.task_path,
#         task_id=Config.data.task_id,
#         task_test_id=Config.data.task_id,
#         w2v_dim=Config.model.embed_dim,
#         use_pretrained=Config.model.use_pretrained
#     )
#     data = data_loader.make_train_and_test_set()
#     train_input, train_question, train_answer, train_input_mask = data['train']
#     vocab_size = len(data_loader.vocab)
#     #(self, embedding_input, input_mask, embedding_question, vocab_size, params)
#     model = DMN_PLUS(train_input, train_input_mask, train_question, vocab_size, Config)
