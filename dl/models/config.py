import os
import torch


class BaseConfig:
    def __init__(self, data_dir):
        self.save_path = os.path.join(data_dir, 'saved_dict', self.get_model_name() + '.ckpt')
        self.log_path = os.path.join(data_dir, 'log', self.model_name)
        self.vocab_path = os.path.join(data_dir, 'data', 'vocab.pkl')

        if not os.path.exists(data_dir + '/saved_dict'):
            os.makedirs(self.save_path)
        self.embeding_path = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_model_name(self):
        return self.model_name

    def set_model_name(self, value):
        self.model_name = value
