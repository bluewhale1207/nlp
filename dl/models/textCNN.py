import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import BaseConfig


class Config(BaseConfig):
    def __init__(self, data_dir):
        self.model_name = 'TextCNN'
        super(Config, self).__init__(data_dir)
        self.dropout = 0.5
        self.vocab_size = 0
        self.num_classes = 0  # 类别数
        self.kernel_lst = (3, 4, 5)  # 卷积核尺寸
        self.filter_num = 256  # 卷积核数量(channels数)
        self.embedding_pretrained = None  # 预训练词向量
        self.embedding_dim = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度

        self.batch_size = 128  # mini-batch大小
        self.learning_rate = 1e-3  # 学习率
        self.num_epochs = 20  # epoch数
        self.log_interval = 100
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练


    def get_model_name(self):
        return self.model_name

    def set_model_name(self, value):
        self.model_name = value


SENTENCE_LIMIT_SIZE = 20  # 句子平均长度


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        chanel_num = 1
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, config.filter_num, (kernel, config.embedding_dim)) for kernel in config.kernel_lst])

        self.fc = nn.Linear(config.filter_num * len(config.kernel_lst), config.num_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X):
        '''
        X: [batch_size, sequence_length]
        '''
        batch_size = X.shape[0]
        embedding_X = self.embedding(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(
            1)  # add channel(=1) [batch_size, channel(=1), sequence_length, embedding_size]
        x = [F.relu(conv(embedding_X)).squeeze(3) for conv in
             self.convs]  # [(batch_size, output_channel, token_num) * len(kernel_sizes)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        out = torch.cat(x, dim=1)
        # flatten = out.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        out = self.dropout(out)
        logit = self.fc(out)
        return logit
