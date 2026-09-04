"""
Taken from https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/confusionmatrix.py
"""

import numpy as np
import torch
import pandas as pd

class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False, device='cpu', lazy=True):
        super().__init__()
        if device == 'cpu':
            self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        else:
            self.conf = torch.zeros((num_classes, num_classes)).cuda()
        self.normalized = normalized
        self.num_classes = num_classes
        self.device = device
        self.reset()
        self.lazy = lazy

    def reset(self):
        if self.device == 'cpu':
            self.conf.fill(0)
        else:
            self.conf = torch.zeros(self.conf.shape).cuda()

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """

        # If target and/or predicted are tensors, convert them to numpy arrays
        if self.device == 'cpu':
            if torch.is_tensor(predicted):
                predicted = predicted.cpu().numpy()
            if torch.is_tensor(target):
                target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if len(predicted.shape) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = predicted.argmax(1)
        else:
            if not self.lazy:
                assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                    'predicted values are not between 0 and k-1'

        if len(target.shape) != 1:
            if not self.lazy:
                assert target.shape[1] == self.num_classes, \
                    'Onehot target does not match size of confusion matrix'
                assert (target >= 0).all() and (target <= 1).all(), \
                    'in one-hot encoding, target values should be 0 or 1'
                assert (target.sum(1) == 1).all(), \
                    'multi-label setting is not supported'
            target = target.argmax(1)
        else:
            if not self.lazy:
                assert (target.max() < self.num_classes) and (target.min() >= 0), \
                    'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target

        if self.device == 'cpu':
            bincount_2d = np.bincount(
                x.astype(np.int64), minlength=self.num_classes ** 2)
            assert bincount_2d.size == self.num_classes ** 2
            conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        else:
            bincount_2d = torch.bincount(
                x, minlength=self.num_classes ** 2)

            conf = bincount_2d.view((self.num_classes, self.num_classes))
        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None, cm_device='cpu', lazy=True):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized, device=cm_device, lazy=lazy)
        self.lazy = lazy
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)

    def get_miou_acc(self, sync=True):
        conf_matrix = self.conf_metric.value()

        # Synchronize confusion matrix across all GPUs in DDP
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.clone()  # avoid in-place mutation of self.conf
            if sync and torch.distributed.is_initialized():
                # All-reduce to sum confusion matrices from all GPUs
                torch.distributed.all_reduce(conf_matrix, op=torch.distributed.ReduceOp.SUM)
            conf_matrix = conf_matrix.cpu().numpy()

        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[true_positive + false_negative == 0] = np.nan  # exclude classes absent from GT
        miou = float(np.nanmean(iou) * 100)
        acc = float(np.diag(conf_matrix).sum() / conf_matrix.sum() * 100)

        return miou, acc

    def get_global_iou(self, sync=True):
        """
        Computes the global IoU (intersection over union) over all classes.

        Global IoU is computed as:
            global_iou = total_true_positives / (2 * total_pixels - total_true_positives)

        Returns:
            float: The global IoU.
        """
        # Get the confusion matrix value
        conf_matrix = self.conf_metric.value()

        # Synchronize confusion matrix across all GPUs in DDP
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.clone()  # avoid in-place mutation of self.conf
            if sync and torch.distributed.is_initialized():
                # All-reduce to sum confusion matrices from all GPUs
                torch.distributed.all_reduce(conf_matrix, op=torch.distributed.ReduceOp.SUM)
            conf_matrix = conf_matrix.cpu().numpy()

        # Optionally ignore the classes specified by ignore_index
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
            
        # Total true positives (sum of the diagonal)
        total_tp = np.diag(conf_matrix).sum()
        # Total number of pixels (all entries in the confusion matrix)
        total_pixels = conf_matrix.sum()
        
        # Compute global IoU while avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            global_iou = total_tp / (2 * total_pixels - total_tp)
            
        return global_iou * 100

    def get_per_class_metrics(self, class_mapping=None, sync=True):
        """
        Compute per-class IoU, Precision, Recall, F1 sorted by number of true pixels.

        Args:
            class_mapping (list of dicts or None): Each dict maps class names -> indices.
                If None, class indices are used as names.
            sync (bool): Whether to synchronize confusion matrix across GPUs.

        Returns:
            pandas.DataFrame with columns: ['Class', '#Pixels', 'IoU', 'Precision', 'Recall', 'F1']
            Sorted by number of true pixels (descending).
        """
        conf_matrix = self.conf_metric.value()

        # Synchronize confusion matrix across all GPUs in DDP
        if torch.is_tensor(conf_matrix):
            conf_matrix = conf_matrix.clone()  # avoid in-place mutation of self.conf
            if sync and torch.distributed.is_initialized():
                # All-reduce to sum confusion matrices from all GPUs
                torch.distributed.all_reduce(conf_matrix, op=torch.distributed.ReduceOp.SUM)
            conf_matrix = conf_matrix.cpu().numpy()

        # Optionally ignore classes
        if self.ignore_index is None:
            keep_idx = list(range(conf_matrix.shape[0]))
        else:
            ignore_idx = self.ignore_index  # tuple of indices
            keep_idx = [i for i in range(conf_matrix.shape[0]) if i not in ignore_idx]
        conf_matrix = conf_matrix[np.ix_(keep_idx, keep_idx)]

        # True Positives, False Positives, False Negatives
        TP = np.diag(conf_matrix)
        FP = np.sum(conf_matrix, axis=0) - TP
        FN = np.sum(conf_matrix, axis=1) - TP

        # Per-class metrics
        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        iou_per_class = TP / (TP + FP + FN + 1e-12)
        f1_per_class = 2 * precision * recall / (precision + recall + 1e-12)

        # Class names
        num_classes = conf_matrix.shape[0]
        if class_mapping is None:
            class_names = [str(i) for i in keep_idx]  # use kept indices
        else:
            mapping_dict = class_mapping[0] if isinstance(class_mapping, list) else class_mapping
            # Only keep class names for indices present in conf_matrix
            class_names = []
            for i in keep_idx:
                # mapping_dict uses 1-based indexing
                name = next((n for n, idx in mapping_dict.items() if idx == i), str(i))
                class_names.append(name)

        # Sort by number of true pixels (row sum)
        row_sums = conf_matrix.sum(axis=1)
        sort_idx = np.argsort(-row_sums)

        data = {
            'Class': [class_names[i] for i in sort_idx],
            '#Pixels': row_sums[sort_idx],
            'IoU': iou_per_class[sort_idx],
            'Precision': precision[sort_idx],
            'Recall': recall[sort_idx],
            'F1': f1_per_class[sort_idx]
        }

        df = pd.DataFrame(data)
        return df

