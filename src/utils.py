import collections.abc
import re
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data

np_str_obj_array_pattern = re.compile(r"[SaUO]")

# Set a global cache
zarr_cache = {}

def pad_tensor(x, l, pad_value=0):
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def pad_collate(batch, pad_value=0):
    # modified default_collate from the official pytorch repo
    # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if len(elem.shape) > 0:
            sizes = [e.shape[0] for e in batch]
            m = max(sizes)
            if not all(s == m for s in sizes):
                # pad tensors which have a temporal dimension
                batch = [pad_tensor(e, m, pad_value=pad_value) for e in batch]
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            # storage = elem.storage()._new_shared(numel)
            # out = elem.new(storage)
            storage = elem.untyped_storage()._new_shared(numel * elem.element_size())
            out = torch.empty((numel,), dtype=elem.dtype, device=elem.device).set_(storage)
            out.resize_(len(batch), *batch[0].shape)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError("Format not managed : {}".format(elem.dtype))

            return pad_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pad_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(pad_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, tuple):  # regular tuple
        transposed = zip(*batch)
        return tuple(pad_collate(samples) for samples in transposed)
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [pad_collate(samples) for samples in transposed]

    raise TypeError("Format not managed : {}".format(elem_type))


def get_ntrainparams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_f1_score(preds, labels, num_classes, ignore_index):
    from sklearn.metrics import f1_score
    # Flatten predictions and labels
    preds_flat = preds.cpu().numpy().flatten()
    labels_flat = labels.cpu().numpy().flatten()

    # Ignore specified index
    valid_mask = labels_flat != ignore_index
    preds_flat = preds_flat[valid_mask]
    labels_flat = labels_flat[valid_mask]

    if len(labels_flat) == 0:
        return 0.0

    # Compute F1 score with zero_division=0 to suppress warnings
    return f1_score(labels_flat, preds_flat, average="macro", labels=np.arange(num_classes), zero_division=0)



def compute_ece(confidences_list, correctness_list, ground_truths, ignore_index, num_bins=15):
    """
    Compute Expected Calibration Error (ECE) for segmentation outputs.
    
    Args:
        confidences_list (list of np.array): List of arrays of per-pixel confidence scores.
        correctness_list (list of np.array): List of arrays of per-pixel correctness flags (1 for correct, 0 for wrong).
        ground_truths (list of np.array): List of arrays of ground-truth labels.
        ignore_index (int): Label value to ignore.
        num_bins (int): Number of bins to use for ECE computation.
        
    Returns:
        float: The computed ECE value.
    """
    if not confidences_list or not correctness_list or not ground_truths:
        return float('nan')
    confidences_all = np.concatenate([arr.flatten() for arr in confidences_list])
    correctness_all = np.concatenate([arr.flatten() for arr in correctness_list])
    ground_truth_all = np.concatenate([gt.flatten() for gt in ground_truths])
    
    # Exclude ignored pixels
    valid_mask = ground_truth_all != ignore_index
    confidences_all = confidences_all[valid_mask]
    correctness_all = correctness_all[valid_mask]
    
    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences_all >= bin_lower) & (confidences_all < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if np.sum(in_bin) > 0:
            avg_confidence_in_bin = np.mean(confidences_all[in_bin])
            avg_accuracy_in_bin = np.mean(correctness_all[in_bin])
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
    return ece

