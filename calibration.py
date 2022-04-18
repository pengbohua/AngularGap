import torch.nn as nn
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class TemperatureScaling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1, dtype=torch.float))

    def forward(self, x):
        x = x * torch.clamp(self.temp, 0.8, 1.2)
        return x


class DiagonalScaling(nn.Module):
    def __init__(self, class_nums):
        super().__init__()
        self.class_nums = class_nums
        self.diag = nn.Parameter(torch.eye(class_nums))
        self.bias = nn.Parameter(torch.zeros(class_nums))

    def forward(self, x):
        x = torch.mm(x, self.diag) + self.bias
        return x


class MatrixScaling(nn.Module):
    def __init__(self, class_nums, off_diagonal_intercept_regularization=True):
        super().__init__()
        self.class_nums = class_nums
        self.odir = off_diagonal_intercept_regularization
        self.mat = torch.nn.Linear(class_nums, class_nums)

    def forward(self, x):
        return self.mat(x)


def calibrationMapping(
    num_cls,
    model,
    val_loader,
    calibration_type="diagonal_scaling",
    calibration_lr=0.01,
    max_iter=10,
    ms_odir=True,
    ms_l=1e-5,
    ms_mu=1e-5,
):
    model.eval()
    logits_list = []
    labels_list = []
    num_classes = num_cls
    criterion = nn.CrossEntropyLoss()
    if calibration_type == "matrix_scaling":
        temp = MatrixScaling(num_classes, ms_odir)
    elif calibration_type == "temperature_scaling":
        temp = TemperatureScaling()
    elif calibration_type == "diagonal_scaling":
        temp = DiagonalScaling(num_classes)
    else:
        raise NotImplementedError
    temp = temp.to(device)
    optimizer = torch.optim.LBFGS(
        temp.parameters(), lr=calibration_lr, max_iter=max_iter
    )

    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            logits, _ = model(data, labels)
            logits_list.append(logits)
            labels_list.append(labels)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

    def closure():
        if calibration_type == "matrix scaling":
            if temp.odir:
                assert (
                    ms_l is not None and ms_mu is not None
                ), "assign l and mu to apply odir regularization"
            regularization = 0
            for i in range(temp.class_nums):
                # off diagonal regularization; bias magnitude regularization
                regularization += ms_l * torch.sum(
                    torch.square(temp.mat.weight[0:i, i])
                )
                regularization += ms_l * torch.sum(
                    torch.square(temp.mat.weight[i + 1 :, i])
                )
                regularization += ms_mu * torch.square(temp.mat.bias[i])
            _logits = temp(logits)
            _loss = criterion(_logits, labels)
        else:
            _loss = criterion(temp(logits), labels)
        _loss.backward()
        return _loss

    optimizer.step(closure)
    return temp


def ece_eval(preds, targets, n_bins=10, bg_cls=0):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    confidences, predictions = (
        confidences[targets > bg_cls],
        predictions[targets > bg_cls],
    )
    accuracies = predictions == targets[targets > bg_cls]
    Bm, acc, conf = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece = 0.0
    bin_idx = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        Bm[bin_idx] = bin_size
        if bin_size > 0:
            accuracy_in_bin = np.sum(accuracies[in_bin])
            acc[bin_idx] = accuracy_in_bin / Bm[bin_idx]
            confidence_in_bin = np.sum(confidences[in_bin])
            conf[bin_idx] = confidence_in_bin / Bm[bin_idx]
        bin_idx += 1

    ece_all = Bm * np.abs((acc - conf)) / Bm.sum()
    ece = ece_all.sum()
    return ece, acc, conf, Bm


def tace_eval(preds, targets, n_bins=10, threshold=1e-4, bg_cls=0):
    init = 0
    if bg_cls == 0:
        init = 1
    preds = preds.astype(np.float32)
    targets = targets.astype(np.float16)
    n_img, n_classes = preds.shape[:2]
    Bm_all, acc_all, conf_all = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    ece_all = []
    for cur_class in range(init, n_classes):
        cur_class_conf = preds[:, cur_class]
        cur_class_conf = cur_class_conf.flatten()
        cur_class_conf_sorted = np.sort(cur_class_conf)
        targets_vec = targets.flatten()
        targets_sorted = targets_vec[cur_class_conf.argsort()]
        # target must be sorted along with cls conf
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        bin_size = len(cur_class_conf_sorted) // n_bins
        ece_cls, Bm, acc, conf = (
            np.zeros(n_bins),
            np.zeros(n_bins),
            np.zeros(n_bins),
            np.zeros(n_bins),
        )
        bin_idx = 0
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = (
                    bin_end_ind - bin_start_ind
                )  # extend last bin until the end of prediction array
            # print('bin start', cur_class_conf_sorted[bin_start_ind])
            # print('bin end', cur_class_conf_sorted[bin_end_ind-1])
            # Bm contains size to compute proportion
            Bm[bin_idx] = bin_size
            # compute bin acc with indices
            bin_acc = targets_sorted[bin_start_ind:bin_end_ind] == cur_class
            acc[bin_idx] = np.sum(bin_acc) / bin_size
            bin_conf = cur_class_conf_sorted[bin_start_ind:bin_end_ind]
            conf[bin_idx] = np.sum(bin_conf) / bin_size
            bin_idx += 1
        # weighted average
        ece_cls = Bm * np.abs((acc - conf)) / (Bm.sum())
        ece_all.append(np.mean(ece_cls))
        Bm_all += Bm
        acc_all += acc
        conf_all += conf
    ece, acc_all, conf_all = (
        np.mean(ece_all),
        acc_all / (n_classes - init),
        conf_all / (n_classes - init),
    )
    return ece, acc_all, conf_all, Bm_all
