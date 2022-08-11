import logging
from collections import Counter

import numpy as np


class Vocab:
    def __init__(self, train_data, label2name):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        self._id2label = []

        reverse_f = lambda x: dict(zip(x, range(len(x))))
        self.build_vocab(train_data, label2name)
        self._word2id = reverse_f(self._id2word)
        self._label2id = reverse_f(self._id2label)
        logging.info("Build vocab: words %d, labels %d." % (self.word_size, self.label_size))

    def build_vocab(self, data, label2name):
        self.word_counter = Counter()
        for text in data:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1

        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        self.label_counter = Counter(data['label'])
        print(self.label_counter)

    def load_pretrained_embs(self, embfile):
        with open(embfile, 'r') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])
        index = len(self._id2extword)
        embeddings = np.zeros(word_count + index, embedding_dim)
        for line in lines[1:]:

    def word2id(self):
        pass

    def id2word(self):
        pass

    def label2id(self):
        pass

    def id2label(self):
        pass

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def label_size(self):
        return len(self._id2label)


vocab = Vocab(train_data)
