import sys
import random
import torch

from sklearn import metrics


class OneshotPredictor:
    def __init__(self, model, train_set, test_set):
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.comparison_tensor = None
        self.comparison_labels = []
        self._sample_train_set()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def _sample_train_set(self):
        labels = set()
        comparison_examples = []

        for i in self.train_set:
            labels.add(i[0])

        for l in labels:
            lab_subset = [i for i in self.train_set if i[0] == l]
            n = 5 if len(lab_subset) > 5 else len(lab_subset)
            samp = random.sample(lab_subset, n)
            comparison_examples.extend([s[1] for s in samp])
            self.comparison_labels.extend([s[0] for s in samp])

        max_len = max([t.shape[0] for t in comparison_examples])
        comparison_tensor = torch.zeros(len(comparison_examples), max_len).long()

        for c_ix, ce in enumerate(comparison_examples):
            comparison_tensor[c_ix, :ce.shape[0]] = ce.unsqueeze(0)

        # print('comparison tensor dtype:', comparison_tensor.dtype)
        print('comparison tensor shape:', comparison_tensor.shape)
        self.comparison_tensor = comparison_tensor

    def oneshot_predict(self, x_input):
        x_tensor = torch.cat([x_input.unsqueeze(0) for _ in range(self.comparison_tensor.shape[0])], dim=0).long()
        # print('new example dtype:', x_tensor.dtype)
        print('new example shape:', x_tensor.shape)
        with torch.no_grad():
            scores = self.model(self.comparison_tensor.to(self.device), x_tensor.to(self.device)).cpu().tolist()

        class_prob_container = list(zip(self.comparison_labels, scores))
        top_class = sorted(class_prob_container, key=lambda x: x[1], reverse=True)[0][0]

        return top_class

    def run_eval(self):
        print('Running eval...')
        y_true, y_pred = [], []
        for ix, test_i in enumerate(self.test_set):
            sys.stdout.write('\rGetting prediction {}...'.format(ix + 1))
            y_hat = self.oneshot_predict(test_i[1])
            y_pred.append(y_hat)
            y_true.append(test_i[0])

        print()
        print(metrics.classification_report(np.asarray(y_true), np.asarray(y_pred)))