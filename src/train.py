import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import DataLoader, RandomSampler

from sklearn import metrics
from data_utils import *
from one_shot_model import *


class Trainer:
    def __init__(self, args, train_data, test_data, vocab_size):
        self.model_path = '/data/users/kyle.shaffer/wm/oneshot_models'
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = args.batch_size
        self.n_epochs = args.n_epochs
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = None
        self.test_loader = None
        self._setup_data()
        self.model = SiameseNet(args, vocab_size=self.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _setup_data(self):
        train_set = DBPediaLoader(self.train_data, few_shot=True)
        test_set = DBPediaLoader(self.test_data, few_shot=False)
        print('Train label distribution:')
        train_set.get_label_freq()
        print('Test label distribution:')
        test_set.get_label_freq()
        train_sampler = RandomSampler(train_set)
        test_sampler = RandomSampler(test_set)
        self.train_loader = DataLoader(train_set, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_set, sampler=test_sampler,
                                       batch_size=self.batch_size, collate_fn=collate_fn)

    def evaluate(self, data):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x_left, x_right, y) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x_left.shape[0]))
            with torch.no_grad():
                x_left, x_right, y = x_left.to(self.device), x_right.to(self.device), y.to(self.device)

                output = self.model(x_left, x_right).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                probs = torch.sigmoid(output).data.cpu().numpy().tolist()
                # print(type(probs))
                preds = [1 if p >= 0.5 else 0 for p in probs]
                y_pred.extend(preds)
                # print(y.data.cpu().numpy().tolist()[:10])
                y_true.extend([int(i) for i in y.data.cpu().numpy().tolist()])

        update_loss = total_loss / data_len
        print('Test loss:', update_loss)
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
        print()

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(args.n_epochs):
            total_loss = 0.0
            self.model.train()
            for x_left, x_right, y_batch in self.train_loader:
                print('=', flush=True, end='')
                x_left, x_right, y_batch = x_left.to(self.device), x_right.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_left, x_right).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader)
            print()

            if test_acc > best_acc:
                print('Saving model...')
                best_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'conv_one_shot_model.pt'))


class ClfTrainer(Trainer):
    def __init__(self, args, train_data, test_data, vocab_size):
        super().__init__(args, train_data, test_data, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.model = ConvClassifier(args=args, vocab_size=vocab_size).to(self.device)

    def _setup_data(self):
        train_set = ClfLoader(self.train_data)
        test_set = ClfLoader(self.test_data)
        print('Train label distribution:')
        train_set.get_label_freq()
        print('Test label distribution:')
        test_set.get_label_freq()
        train_sampler = RandomSampler(train_set)
        test_sampler = RandomSampler(test_set)
        self.train_loader = DataLoader(train_set, sampler=train_sampler,
                                       batch_size=self.batch_size, collate_fn=clf_collate_fn)
        self.test_loader = DataLoader(test_set, sampler=test_sampler,
                                       batch_size=self.batch_size, collate_fn=clf_collate_fn)

    def evaluate(self, data):
        self.model.eval()

        data_len = len(data)
        total_loss = 0.0
        y_true, y_pred = [], []

        for ix, (x, y) in enumerate(data):
            # print('*', flush=True, end='')
            sys.stdout.write('\rEvaluating {} examples...'.format((ix + 1) * x.shape[0]))
            with torch.no_grad():
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x).squeeze()
                # print('output:', output.shape)
                losses = self.criterion(output, y)

                total_loss += losses.item() # .data[0]
                # print(type(probs))
                preds = torch.argmax(output, dim=1).cpu().tolist()
                y_pred.extend(preds)
                # print(y.data.cpu().numpy().tolist()[:10])
                y_true.extend([int(i) for i in y.data.cpu().numpy().tolist()])

        update_loss = total_loss / data_len
        print('Test loss:', update_loss)
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))
        print()

        return metrics.accuracy_score(np.asarray(y_true), np.asarray(y_pred))

    def train(self):
        best_acc = 0.0
        for epoch in range(args.n_epochs):
            total_loss = 0.0
            self.model.train()
            for x_batch, y_batch in self.train_loader:
                print('=', flush=True, end='')
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            print('\nSummary for Epoch {}'.format(epoch + 1))
            print('Train Loss:', total_loss / len(self.train_loader))
            print('Train Metrics')
            train_acc = self.evaluate(self.train_loader)
            print('Test Metrics')
            test_acc = self.evaluate(self.test_loader)
            print()

            if test_acc > best_acc:
                print('Saving model...')
                best_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.model_path, 'db_clf.pt'))


def main(args):
    data_formatter = DataFormatter()
    train_data, test_data = data_formatter.get_data()
    print(len(train_data))
    print(len(test_data))
    if args.task == 'oneshot':
        print('Training one-shot model...')
        trainer = Trainer(args, train_data, test_data, data_formatter.vocab_size)
    else:
        print('Training classifier...')
        trainer = ClfTrainer(args, train_data, test_data, data_formatter.vocab_size)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data params
    parser.add_argument('--gpu', type=int, required=False, default=0)
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--n_epochs', type=int, required=False, default=10)
    parser.add_argument('--task', type=str, required=False, default='oneshot')
    # Model params
    parser.add_argument('--embed_dim', type=int, required=False, default=200)
    parser.add_argument('--hidden_dim', type=int, required=False, default=256)
    parser.add_argument('--hidden_layers', type=int, required=False, default=3)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    main(args)
