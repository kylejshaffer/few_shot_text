import os
import numpy as np
import pandas as pd

import torch
from data_utils import *
from infer_utils import *
from one_shot_model import *


class DummyArgs:
    embed_dim = 200
    hidden_dim = 256
    hidden_layers = 3


def run():
    data_formatter = DataFormatter()
    train, test = data_formatter.get_data()
    print('Train distirubtion')
    print(pd.Series([i[0] for i in train]).value_counts())
    print('Test distribution')
    print(pd.Series([i[0] for i in test]).value_counts())

    args = DummyArgs()

    model_path = '/data/users/kyle.shaffer/wm/oneshot_models/stacked_one_shot_model.pt'
    model = SiameseNet(vocab_size=data_formatter.vocab_size, args=args)
    model.load_state_dict(torch.load(model_path))
    print('Model loaded...')

    predictor = OneshotPredictor(model=model, train_set=train, test_set=test[:10000])
    print( predictor.comparison_labels)
    predictor.run_eval()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    run()