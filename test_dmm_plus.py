from hbconfig import Config
import os
from utils.data_loader import DataLoader
from model.dmn_plus import DMN_PLUS


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
    params_path = os.path.join(config_path, 'bAbi_task1.yml')
    Config(params_path)
    print("Config: ", Config)
    print(params_path)
    data_loader = DataLoader(
        task_path=Config.data.task_path,
        task_id=Config.data.task_id,
        task_test_id=Config.data.task_id,
        w2v_dim=Config.model.embed_dim,
        use_pretrained=Config.model.use_pretrained
    )
    data = data_loader.make_train_and_test_set()
    train_input, train_question, train_answer, train_input_mask = data['train']
    vocab_size = len(data_loader.vocab)
    #(self, embedding_input, input_mask, embedding_question, vocab_size, params)
    model = DMN_PLUS(train_input, train_input_mask, train_question, vocab_size, Config)