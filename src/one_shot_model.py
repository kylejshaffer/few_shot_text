import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNet(nn.Module):
    def __init__(self, args, vocab_size:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.hidden_layers = args.hidden_layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_dropout = nn.Dropout(p=0.2)
        self.dense_dropout = nn.Dropout(p=0.2)
        self.encoders = nn.ModuleList([nn.Conv1d(in_channels=self.embed_dim, out_channels=self.hidden_dim, kernel_size=3, stride=1),
                                       nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=1),
                                       nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, stride=1)])
        self.pool = nn.AvgPool1d(2)
        # self.dense_hidden = nn.Linear(in_features=int(2 * (self.hidden_dim * len(self.encoders))), out_features=512)
        self.dense_hidden = nn.Linear(in_features=int(2 * self.hidden_dim), out_features=512)
        self.dense_out = nn.Linear(in_features=512, out_features=1)

    def _encode_and_pool(self, x, encoder):
        return torch.relu(self.pool(encoder(x)))

    def forward(self, x_left, x_right):
        x_left_embed = self.embed_dropout(self.embed(x_left)).permute(0, 2, 1)
        x_right_embed = self.embed_dropout(self.embed(x_right)).permute(0, 2, 1)

        # Stacked left encoder output
        for e_ix, e in enumerate(self.encoders):
            if e_ix == 0:
                x_left = self._encode_and_pool(x=x_left_embed, encoder=e)
            else:
                x_left = self._encode_and_pool(x=x_left, encoder=e)

        # Stacked right encoder output
        for e_ix, e in enumerate(self.encoders):
            if e_ix == 0:
                x_right = self._encode_and_pool(x=x_right_embed, encoder=e)
            else:
                x_right = self._encode_and_pool(x=x_right, encoder=e)

        # Pooling
        x_right = x_right.max(2)[0]
        x_left = x_left.max(2)[0]

        # final_encoded = x_left - x_right
        minus = x_left - x_right
        plus = x_left + x_right
        final_encoded = torch.cat([torch.abs(minus), plus], 1)

        hidden_out = torch.relu(self.dense_hidden(final_encoded))
        logits_out = self.dense_out(self.dense_dropout(hidden_out))

        return logits_out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, required=False, default=10000)
    parser.add_argument('--hidden_dim', type=int, required=False, default=256)
    parser.add_argument('--hidden_layers', type=int, required=False, default=3)
    parser.add_argument('--embed_dim', type=int, required=False, default=128)
    args = parser.parse_args()

    net = SiameseNet(args)
    print(net)

    x_left = torch.zeros(size=(1, 10), dtype=torch.long)
    x_right = torch.zeros(size=(1, 10), dtype=torch.long)

    out = net(x_left, x_right)
    print(out.shape)
