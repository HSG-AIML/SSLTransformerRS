import numpy as np
from collections import defaultdict


class ClasswiseAccuracy(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp_per_class = defaultdict(int)
        self.count_per_class = defaultdict(int)
        self.count = 0
        self.tp = 0

    def add_batch(self, y, y_hat):
        for true, pred in zip(y, y_hat):
            self.count_per_class["class_" + str(true.item())] += 1
            self.count += 1
            if true == pred:
                self.tp_per_class["class_" + str(true.item())] += 1
                self.tp += 1

    def get_classwise_accuracy(self):
        return {k: self.tp_per_class[k] / count for k, count in self.count_per_class.items()}

    def get_average_accuracy(self):
        cw_acc = self.get_classwise_accuracy()
        return np.mean(list(cw_acc.values()))

    def get_overall_accuracy(self):
        return self.tp / self.count


class ClasswiseMultilabelMetrics(object):
    def __init__(self, num_classes, prefix="class_"):
        self.num_classes = num_classes
        self.prefix = prefix
        self.data = {prefix + str(i): {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for i in range(num_classes)}  # holds the classwise tp,tn,fp,fn

        # hold the overall tp,tn,fp,fn
        self.num_tp = 0
        self.num_fp = 0
        self.num_tn = 0
        self.num_fn = 0

    def add_batch(self, y_batch, y_hat_batch):
        for y, y_hat in zip(y_batch, y_hat_batch):
            for i in range(self.num_classes):
                class_data = self.data[self.prefix + str(i)]
                if y_hat[i] == y[i]:
                    # true-x
                    if y_hat[i]:
                        class_data["tp"] += 1
                        self.num_tp += 1
                    else:
                        class_data["tn"] += 1
                        self.num_tn += 1
                else:
                    # false-x
                    if y_hat[i]:
                        class_data["fp"] += 1
                        self.num_fp += 1
                    else:
                        class_data["fn"] += 1
                        self.num_fn += 1

    def get_classwise_precision(self):
        # tp / (tp + fp)
        out = {}
        for i in range(self.num_classes):
            class_data = self.data[self.prefix + str(i)]
            if class_data["tp"] == 0:
                out[self.prefix + str(i)] = 0
            else:
                out[self.prefix + str(i)] = class_data["tp"] / (class_data["tp"] + class_data["fp"])

        return out

    def get_classwise_recall(self):
        # tp / (tp + fn)
        out = {}
        for i in range(self.num_classes):
            class_data = self.data[self.prefix + str(i)]
            if class_data["tp"] == 0:
                out[self.prefix + str(i)] = 0
            else:
                out[self.prefix + str(i)] = class_data["tp"] / (class_data["tp"] + class_data["fn"])

        return out

    def get_classwise_f1(self):
        # 2 * (precision * recall) / (precision + recall)
        cw_prec = self.get_classwise_precision()
        cw_rec = self.get_classwise_recall()

        out = {}
        for i in range(self.num_classes):
            precision = cw_prec[self.prefix + str(i)]
            recall = cw_rec[self.prefix + str(i)]

            if (precision + recall) == 0:
                out[self.prefix + str(i)] = 0
            else:
                out[self.prefix + str(i)] = 2 * (precision * recall) / (precision + recall)

        return out

    def get_average_f1(self):
        cw_f1 = self.get_classwise_f1()

        return np.mean(list(cw_f1.values()))

    def get_average_recall(self):
        cw_recall = self.get_classwise_recall()

        return np.mean(list(cw_recall.values()))

    def get_average_precision(self):
        cw_precision = self.get_classwise_precision()

        return np.mean(list(cw_precision.values()))

    def get_overall_precision(self):
        # tp / (tp + fp)
        if self.num_tp == 0:
            return 0
        else:
            return self.num_tp / (self.num_tp + self.num_fp)

    def get_overall_recall(self):
        # tp / (tp + fn)
        if self.num_tp == 0:
            return 0
        else:
            return self.num_tp / (self.num_tp + self.num_fn)

    def get_overall_f1(self):
        # 2 * (precision * recall) / (precision + recall)
        precision = self.get_overall_precision()
        recall = self.get_overall_recall()

        if (precision + recall) == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)


class PixelwiseMetrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.count = 0

        self.data = {"pixelclass_" + str(i): {"acc": 0} for i in range(num_classes)}

    def add_batch(self, y, y_hat):
        self.count += 1

        for c in range(self.num_classes):
            class_data = self.data["pixelclass_" + str(c)]
            preds_c = y_hat == c
            targs_c = y == c
            num_correct = (preds_c * targs_c).sum().cpu().detach().numpy()
            num_pixels = np.sum(targs_c.cpu().detach().numpy())
            class_data["acc"] += num_correct / num_pixels

    def get_classwise_accuracy(self):
        return {k: el['acc'] / self.count for k, el in self.data.items()}

    def get_average_accuracy(self):
        cw_acc = self.get_classwise_accuracy()
        return np.mean(list(cw_acc.values()))
