import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
import numpy as np


def angular_gap(cos_dists, label_onehot, posterior=None):

    batch_size, num_cls = cos_dists.shape
    if posterior is None:
        posterior = torch.ones(1, num_cls, device=label_onehot.device)
    targets_cosine = torch.sum(cos_dists * label_onehot, 1)
    min_excl_cosine, min_angle_excl_idx = torch.max(
        cos_dists
        * (torch.ones_like(label_onehot, device=label_onehot.device) - label_onehot),
        dim=1,
    )

    cos_margin = targets_cosine - min_excl_cosine
    targets_angle = torch.acos(torch.clamp(targets_cosine, -1.0 + 1e-7, 1.0 - 1e-7))
    min_excl_angle = torch.acos(torch.clamp(min_excl_cosine, -1.0 + 1e-7, 1.0 - 1e-7))
    s_max_excl = (
        posterior.expand(batch_size, num_cls)
        .gather(1, min_angle_excl_idx[:, None])
        .squeeze()
    )
    s_y = torch.sum(posterior.expand(batch_size, num_cls) * label_onehot, 1)

    angular_margin = targets_angle * s_y - min_excl_angle * s_max_excl
    return targets_cosine, cos_margin, targets_angle, angular_margin


def avh(cosine_dists, targets):
    """'
    @param cosine_dists: B x C
    @param targets: C
    @return:
    """
    ang_dists = torch.acos(torch.clamp(cosine_dists, -1.0 + 1e-7, 1.0 - 1e-7))
    avh = (
        ang_dists.gather(1, targets[:, None]) / ang_dists.sum(1, keepdim=True).squeeze()
    )
    return avh


def get_confidence_output_margin(logits, label_onehot):
    confidence = F.softmax(logits, 1)
    targets_conf = torch.sum(confidence * label_onehot, 1)

    max_excl_conf, _ = torch.max(
        confidence
        * (torch.ones_like(label_onehot, device=label_onehot.device) - label_onehot),
        dim=1,
    )
    conf_output_margin = targets_conf - max_excl_conf
    return targets_conf, conf_output_margin


def embed_norm(cls_embeddings: Tensor) -> Tensor:
    """

    @param cls_embeddings: N (C) x F
    @return: l2 norm as a scalar
    """
    return torch.norm(cls_embeddings, dim=1)


def get_kendalltau(score1, score2, standard_order=True):
    if standard_order:
        return kendalltau(score1, score2)
    elif isinstance(score1, dict) and isinstance(score2, dict):
        res = np.zeros(len(score1), 2)
        for (k0, v0), (k1, v1) in zip(score1.items(), score2.items()):
            res[k0][0] = v1
            res[k1][1] = v1
        return kendalltau(res[:, 0], res[:, 1])


def get_spearman(score1, score2, standard_order=True):
    """
    eq: 1 - 6*sum((s1-s2)**2)/(n*(n**2-1))
    @param cls_embeddings: N x F
    @return: l2 norm as a scalar
    """
    if standard_order:
        return spearmanr(score1, score2)
    elif isinstance(score1, dict) and isinstance(score2, dict):
        res = np.zeros(len(score1), 2)
        for (k0, v0), (k1, v1) in zip(score1.items(), score2.items()):
            res[k0][0] = v1
            res[k1][1] = v1
        return spearmanr(res[:, 0], res[:, 1])
