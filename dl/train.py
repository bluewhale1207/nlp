import os
import time
import logging
from importlib import import_module
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from tensorboardX import SummaryWriter

from data_process import build_vocab, tokenizer, build_iterator, read_category, UNK
from settings import ROOT_DIR

clip = 5.0
epochs = 1
early_stops = 3
log_interval = 50

save_model = '{model_name}_{epoch_num}.bin'
save_test_file = '{model_name}.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择 cuda
print(device)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# class Config:
#     def __init__(self, learning_rate):
#         self.learning_rate = 0.1
#         self.num_epochs = 10
#         self.log_path = ''
#         self.log_interval = 100
#
#
# config = Config()


class Trainer:
    def __init__(self, model, train_path, val_path, test_path):
        self.model = model
        self.batch_size = 128
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        self.vocab_map = vocab_map
        self.labels_map = labels_map

        self.train_iter = self.build_dataset(self.train_path)
        self.dev_iter = self.build_dataset(self.val_path)
        self.test_iter = self.build_dataset(self.test_path)

        self.criterion = nn.CrossEntropyLoss()  # F.cross_entropy
        self.optimizer = None
        self.cur_step = 0
        self.best_train_f1, self.best_dev_f1 = 0, 0
        self.target_names = []
        self.step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_dataset(self, filepath):
        contents = []
        labels = []
        i = 0
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    label, content = line.split('\t')
                    content_ = [self.vocab_map.get(i, self.vocab_map.get(UNK)) for i in tokenizer(content)]
                    contents.append(content_)
                    labels.append(self.labels_map[label])
                except Exception as e:
                    print(e)
                    print(line)
                i += 1

                if i > 1000:
                    break
        data = build_iterator(contents, labels, self.batch_size)
        return data

    def train(self):
        logging.info('Start training...')
        start_time = time.time()
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
        for epoch in range(config.num_epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
            # scheduler.step() # 学习率衰减
            for i, (trains, labels) in enumerate(self.train_iter):
                outputs = model(trains)
                model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                if total_batch % 100 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    f1 = metrics.f1_score(true, predic, average='micro')
                    dev_acc, dev_loss = self.evaluate(config, model, self.dev_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    model.train()
                total_batch += 1
                if total_batch - last_improve > config.require_improvement:
                    # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
            if flag:
                break
        writer.close()
        self.test(config, model, self.test_iter)

    def test(self, config, model, test_iter):
        # test
        model.load_state_dict(torch.load(config.save_path))
        model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(config, model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        print(msg.format(test_loss, test_acc))
        print("Precision, Recall and F1-Score...")
        print(test_report)
        print("Confusion Matrix...")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

    def evaluate(self, config, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in data_iter:
                outputs = model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)

    # def _train(self, epoch):
    #     self.optimizer.zero_grad()
    #     self.model.train()
    #     start_time = time.time()
    #     total_losses = 0
    #     losses = 0
    #     y_pred = []
    #     y_true = []
    #
    #     for batch_idx, batch_data in enumerate(self.train_data):
    #         torch.cuda.empty_cache()
    #         batch_inputs, batch_labels = self.batch2tensor(batch_data)
    #         batch_preds = self.model(batch_inputs)
    #         loss = self.criterion(batch_preds, batch_labels)
    #         loss.backward()
    #         loss_val = loss.detch().cpu().item()
    #         total_losses += loss_val
    #         losses += loss_val
    #         output_labels = torch.max(batch_preds, dim=1)[1]
    #         y_pred.extend(output_labels.cpu().numpy().to_list())
    #         y_true.extend(batch_labels.cpu().numpy().to_list())
    #
    #         nn.utils.clip_grad_norm(self.optimizer.all_params, max_norm=clip)
    #         for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
    #             optimizer.step()
    #             scheduler.step()
    #         self.optimizer.zero_grad()
    #         if batch_idx % log_interval == 0:
    #             elapsed = time.time() - start_time
    #             lrs = self.optimizer.get_lr()
    #
    #             logging.info(
    #                 '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr {} | loss {:.4f} | s/batch {:.2f}'.format(
    #                     epoch, self.step, batch_idx, self.batch_num, lrs,
    #                     losses / log_interval,
    #                     elapsed / log_interval))
    #
    #     during_time = time.time() - start_time
    #
    # def test(self):
    #     logging.info('Start test...')
    #     self.model.load_state_dict(torch.load(save_model))
    #     self._eval(self.last_epoch + 1, test=True)

    def _eval(self, epoch, test=False):
        self.model.eval()
        start_time = time.time()
        data = self.test_data if test else self.dev_data
        y_pred = []
        y_true = []

        with torch.no_grad():
            for i, batch_data in enumerate(data):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)
                batch_outputs = self.model(batch_inputs)
                output_labels = torch.max(batch_outputs, dim=1)[1]
                y_pred.extend(output_labels.cpu().numpy().to_list())
                y_true.extend(batch_labels.cpu().numpy().to_list())
            f1 = get_score(y_true, y_pred)
            during_time = time.time() - start_time

            if test:
                df = pd.DataFrame({'label': y_pred, 'data': batch_data})
                df.to_csv(save_test_file)
            else:
                logging.info(
                    '| epoch {:3d} | dev | score {} | f1 {} | time {:.2f}'.format(epoch, f1, during_time))
                report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
                logging.info('\n' + report)

    def batch2tensor(self):
        pass


if __name__ == "__main__":
    model_name = 'textCNN'
    base_dir = os.path.join(ROOT_DIR, 'data', 'cnews')
    save_dir = os.path.join(ROOT_DIR, 'models')
    train_path = os.path.join(base_dir, 'cnews.train.txt')
    test_path = os.path.join(base_dir, 'cnews.test.txt')
    val_path = os.path.join(base_dir, 'cnews.val.txt')
    vocab_path = os.path.join(base_dir, 'cnews.pkl')
    # save_path = os.path.join(save_dir, 'best_validation')

    min_freq = 1
    is_word = True
    vocabs = build_vocab(tokenizer, train_path, min_freq, is_word)
    labels_map = read_category(test_path)
    vocab_map = vocabs.get_stoi()

    batch_size = 128
    x = import_module('models.' + model_name)
    config = x.Config(save_dir)
    config.vocab_size = len(vocab_map)
    config.num_classes = len(labels_map)

    model = x.Model(config)
    Trainer(model, train_path, test_path, val_path).train()

# for epoch in range(0, config.num_epochs + 1):
#     print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
#
#     for batch_idx, (batch_inputs, batch_labels) in enumerate(self.train_iter):
#         torch.cuda.empty_cache()
#         batch_preds = self.model(batch_inputs)
#         self.model.zero_grad()
#
#         loss = self.criterion(batch_preds, batch_labels)
#         loss_val = loss.detach().cpu().item()
#         output_labels = torch.max(batch_preds, dim=1)[1].cpu()
#         y_pred.extend(output_labels.cpu().numpy().to_list())
#         y_true.extend(batch_labels.cpu().numpy().to_list())
#         nn.utils.clip_grad_norm(self.optimizer.all_params, max_norm=clip)
#         epoch_loss += loss.item()
#
#         loss.backward()
#         optimizer.step()
#         if epoch % config.log_interval == 0:
#             true = batch_labels.data.cpu()
#             train_acc = metrics.accuracy_score(true, output_labels)
#             # dev_acc, dev_loss = evaluate(config, model, dev_iter)
#             corrects = (true == output_labels).sum()
#             print(
#                 '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,
#                                                                          loss.item(),
#                                                                          train_acc,
#                                                                          corrects,
#                                                                          true.shape[0]))
#             elapsed = time.time() - start_time
#             lrs = self.optimizer.get_lr()
#
#             logging.info(
#                 '| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr {} | loss {:.4f} | s/batch {:.2f} acc: {:.4f}%({}/{})'.format(
#                     epoch, self.step, batch_idx, self.batch_size, lrs,
#                     epoch_loss / log_interval,
#                     elapsed / log_interval), train_acc,
#                 corrects,
#                 true.shape[0])
#             # train_f1 = self._train(epoch)
#             dev_f1 = self._eval(epoch)
#             if self.best_dev_f1 < dev_f1:
#                 self.best_train_f1 = dev_f1
#                 torch.save(self.model.state_dict(), save_model)
#                 self.best_train_f1 = train_f1
#                 self.early_stop = 0
#             else:
#                 self.early_stop += 1
#                 if self.early_stop == early_stops:
#                     logging.info(
#                         "Eearly stop in epoch %d, best train: %.2f, dev: %.2f" % (
#                             epoch, self.best_train_f1, self.best_dev_f1))
#                     self.last_epoch = epoch
#                     break
