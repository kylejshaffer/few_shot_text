import io
import json
import os
import random
import sys
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torchtext
import torch.nn.functional as F
from torchtext.utils import unicode_csv_reader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import text_classification
from torch.utils.data import Dataset, DataLoader, RandomSampler


class DataFormatter:
    def __init__(self):
        self.labs_to_sample = [2, 5, 13]
        self.data_path = '/data/users/kyle.shaffer/pytorch_data'
        self.n_train_samp = 50000
        self.n_test_samp = 20000
        self.n_shot = 2
        self.word2id = None
        self.tok_set = None
        self.vocab_size = None
        self._setup_tok_ids()

    def _setup_tok_ids(self):
        with open('/home/kyle.shaffer/proj-wm/oneshot-category-analytic/word2id_thresh10.json', mode='r') as infile:
            word2id = json.load(infile)

        self.word2id = word2id
        self.tok_set = set(word2id.keys())
        self.vocab_size = len(self.word2id)

    def _filter_tok_ids(self, input_):
        new_toks = [self.word2id[t] if (t in self.tok_set) else self.word2id['<unk>'] for t in input_]
        return torch.Tensor(new_toks).long()

    def get_datalines(self, data_path:str, tokenizer):
        data = []
        with io.open(data_path, encoding='utf8', mode='r') as infile:
            reader = unicode_csv_reader(infile)
            for row_ix, row in enumerate(reader):
                tokens = ' '.join(row[1:])
                tokens = tokenizer(tokens)
                sys.stdout.write('\rProcessing line {}...'.format(row_ix))
                tok_ids = self._filter_tok_ids(tokens)
                lab = int(row[0]) - 1
                data.append((lab, tok_ids))
        # Hack for log formatting
        print()

        return data

    def get_data(self):
        if not(os.path.exists(self.data_path)):
            print('Downloading data...')
            os.mkdir(self.data_path)

            train_obj, test_obj = text_classification.DATASETS['DBpedia'](root=self.data_path, vocab=None)
            train = train_obj._data
            test = test_obj._data
        else:
            print('Reading pre-saved data...')
            tokenizer = get_tokenizer("basic_english")
            train = self.get_datalines(os.path.join(self.data_path, 'dbpedia_csv/train.csv'), tokenizer)
            test = self.get_datalines(os.path.join(self.data_path, 'dbpedia_csv/test.csv'), tokenizer)

        random.shuffle(train)
        random.shuffle(test)
        train = train[:self.n_train_samp]
        test = test[:self.n_test_samp]

        few_shot_examples = []
        # Get sampled examples for few-shot case
        for l in self.labs_to_sample:
            samp = [i for i in train if i[0] == l][:self.n_shot]
            few_shot_examples.extend(samp)
        train = [i for i in train if not(i[0] in set(self.labs_to_sample))]
        train.extend(few_shot_examples)

        return train, test


class DBPediaLoader(Dataset):
    def __init__(self, data, few_shot:bool=True):
        self.labs_to_sample = [2, 5, 13] # hard-coded for now...
        self.data = data
        self.few_shot = few_shot
        self.lab_to_data = None
        self._get_lab_data_maps()
        self.pair_data = self._build_pairs()

    def get_label_freq(self):
        print(pd.Series([i['label'] for i in self.pair_data]).value_counts())

    def _get_lab_data_maps(self):
        lab_data_map = {}
        for d in self.data:
            k_d, v_d = d
            if not(k_d in lab_data_map.keys()):
                lab_data_map[k_d] = []
            lab_data_map[k_d].append(v_d)

        self.lab_to_data = lab_data_map

    def _build_pairs(self):
        n = 10000
        examples = []

        if self.few_shot:
            # Match pairs with `n`-shot samples
            large_labs = [l for l in self.lab_to_data.keys() if not(l in self.labs_to_sample)]
            print('Matching with n-shot samples')
            for l in self.labs_to_sample:
                l_samp = [i for i in self.data if i[0] == l]
                example_dict = {'left': l_samp[0][1], 'right': l_samp[1][1], 'label': 1}
                examples.append(example_dict)
                for k, v in self.lab_to_data.items():
                    if not(k in self.labs_to_sample):
                        samp = random.sample(v, int(n / 60))
                        for l_i in l_samp:
                            for l_j in samp:
                                example_dict = {'left': l_i[1],
                                                'right': l_j,
                                                'label': 0}
                                examples.append(example_dict)
                                # print('X size:', len(X))
        else:
            large_labs = list(self.lab_to_data.keys()) # [l for l in self.lab_to_data.keys() if not(l in self.labs_to_sample)]

        # Match pairs from other classes
        print('Getting other matches')
        for ll in large_labs:
            c = 0
            rand_indices = list(range(len(self.lab_to_data[ll])))
            random.shuffle(rand_indices)
            for t_i, t in enumerate(self.lab_to_data[ll]):
                # Sample positive pair example
                if c > int(n / 3):
                    break
                if t_i == rand_indices[t_i]:
                    continue
                example_dict = {'left': t,
                                'right': self.lab_to_data[ll][rand_indices[t_i]],
                                'label': 1}
                examples.append(example_dict)
                # print('X size:', len(X))
                c += 1

                # Sample negative pair
                rand_lab_match = random.sample([l for l in large_labs if l != ll], 1)[0]
                rand_neg_samp = random.sample(self.lab_to_data[rand_lab_match], 1)[0]
                example_dict = {'left': t,
                                'right': rand_neg_samp,
                                'label': 0}
                examples.append(example_dict)

        return examples

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, index):
        return self.pair_data[index]


def collate_fn(batch):
    max_len = 150
    left_lens = [i['left'].shape[0] for i in batch]
    right_lens = [i['right'].shape[0] for i in batch]
    bsz, max_right_len, max_left_len = len(batch),  max(right_lens), max(left_lens)

    left_tensor = torch.zeros(bsz, max_left_len, dtype=torch.long)
    right_tensor = torch.zeros(bsz, max_right_len, dtype=torch.long)
    label_tensor = torch.Tensor([i['label'] for i in batch])

    for b_ix, b in enumerate(batch):
        left_tensor[b_ix, :b['left'].shape[0]] = b['left'].unsqueeze(0)
        right_tensor[b_ix, :b['right'].shape[0]] = b['right'].unsqueeze(0)

    return left_tensor, right_tensor, label_tensor

def load_dataset(data):
    dataset = DBPediaLoader(data)
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler)


if __name__ == '__main__':
    train, test = DataFormatter().get_data()
    print('Original Label Distributions')
    print('Train distirubtion')
    print(pd.Series([i[0] for i in train]).value_counts())
    print('Test distribution')
    print(pd.Series([i[0] for i in test]).value_counts())
    train_set = DBPediaLoader(train, few_shot=True)
    print(len(train_set))
    test_set = DBPediaLoader(test, few_shot=False)
    print(len(test_set))
    print('Binary Label Distributions')
    print('Train binary distribution')
    train_set.get_label_freq()
    print('Test binary distribution')
    test_set.get_label_freq()
    # data_sampler = RandomSampler(dataset)
    # data_loader = DataLoader(dataset, sampler=data_sampler, batch_size=64, collate_fn=collate_fn)
    # print(dir(data_loader))

    # for batch in data_loader:
    #     print(batch[0].shape, batch[1].shape, batch[2].shape)
