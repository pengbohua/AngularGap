import torch
import torch.nn as nn
from transfer_losses import TransferLoss, DFTransferLoss
import backbones
from apex import amp
import torch.nn.functional as F
from torch import Tensor


class TransferNet(nn.Module):
    def __init__(
        self,
        num_class,
        base_net="resnet50",
        transfer_loss="lmmd",
        use_bottleneck=True,
        bottleneck_width=256,
        max_iter=1000,
        **kwargs
    ):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        self.base_network = backbones.get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.amp = None
        self.tr_weight = 10
        self.optimizer = None
        self.lr_scheduler = None
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU(),
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": "lmmd",
            "max_iter": max_iter,
            "num_class": num_class,
        }
        # self.adapt_loss = TransferLoss(**transfer_loss_args)
        transfer_loss_args["k"] = nn.parameter.Parameter(
            torch.ones(1, device="cuda") * 10
        )
        self.adapt_loss = DFTransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.mmd = MMDLoss()
        self.gk1 = GaussianKernel(1.0, 1.0)
        self.gk5 = GaussianKernel(1.0, 5.0)

    def forward(self, source, target, source_label):
        source = self.base_network(source)
        target = self.base_network(target)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
        # classification
        source_clf = self.classifier_layer(source)
        src_target_mask = torch.zeros(len(source_label), 31, device=source.device)
        src_target_mask = src_target_mask.scatter(
            dim=1,
            index=source_label[:, None],
            src=torch.ones((len(source_label), 1), device=source.device),
        )  # B, C
        _, src_conf_margin = get_confidence_output_margin(source_clf, src_target_mask)

        clf_loss = self.criterion(source_clf, source_label)

        # transfer
        kwargs = {}
        kwargs["source_label"] = source_label
        target_clf = self.classifier_layer(target)
        kwargs["target_logits"] = torch.nn.functional.softmax(target_clf, dim=1)

        with torch.no_grad():
            curr_mmd = self.mmd(source, target)
            curr_gk1 = self.gk1(source, target).mean()
            curr_gk5 = self.gk5(source, target).mean()

        tar_pseudo_labels = target_clf.argmax(1)
        tar_target_mask = torch.zeros(len(tar_pseudo_labels), 31, device=target.device)
        tar_target_mask = tar_target_mask.scatter(
            dim=1,
            index=tar_pseudo_labels[:, None],
            src=torch.ones(len(tar_pseudo_labels), 1, device=target.device),
        )
        _, tar_conf_margin = get_confidence_output_margin(target_clf, tar_target_mask)

        kwargs['src_conf_margin'] = src_conf_margin
        transfer_loss, _, _, _ = self.adapt_loss(source, target, **kwargs)
        loss = clf_loss + transfer_loss * self.tr_weight
        if self.amp:
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
        else:
            pass
        return clf_loss, transfer_loss, curr_mmd, curr_gk1, curr_gk5

    def get_parameters(self, initial_lr=1.0):
        params = [
            {"params": self.base_network.parameters(), "lr": 0.1 * initial_lr},
            {"params": self.classifier_layer.parameters(), "lr": 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {"params": self.bottleneck_layer.parameters(), "lr": 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {
                    "params": self.adapt_loss.loss_func.domain_classifier.parameters(),
                    "lr": 1.0 * initial_lr,
                }
            )
        elif self.transfer_loss == "daan":
            params.append(
                {
                    "params": self.adapt_loss.loss_func.domain_classifier.parameters(),
                    "lr": 1.0 * initial_lr,
                }
            )
            params.append(
                {
                    "params": self.adapt_loss.loss_func.local_classifiers.parameters(),
                    "lr": 1.0 * initial_lr,
                }
            )

        return params

    def predict(self, x):
        features = self.base_network(x)
        x = self.bottleneck_layer(features)
        clf = self.classifier_layer(x)
        return clf

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass


def get_confidence_output_margin(logits, label_mask):
    confidence = F.softmax(logits, 1)
    targets_conf = torch.sum(confidence * label_mask, 1)

    max_excl_conf, _ = torch.max(
        confidence * (torch.ones_like(label_mask, device=logits.device) - label_mask),
        dim=1,
    )
    conf_output_margin = targets_conf - max_excl_conf
    return targets_conf, conf_output_margin


class MMDLoss(nn.Module):
    def forward(self, src_feats: Tensor, tar_feats: Tensor) -> Tensor:
        delta = src_feats.mean(0) - tar_feats.mean(0)
        loss = delta.dot(delta.t())
        return loss


class GaussianKernel(nn.Module):
    def __init__(self, amplitude: float, sigma: float):
        super(GaussianKernel, self).__init__()
        self.amplitude = amplitude
        self.sigma = sigma

    def get_covariance_matrix(self, src_feats: Tensor, tar_feats: Tensor) -> Tensor:
        """
        :param src_feats: src_batch_size x F
        :param tar_feats: tar_batch_size x F
        :return: Tensor of src_batch_size x tar_batch_size
        """

        distances_array = torch.stack(
            [
                torch.stack([torch.linalg.norm(x_p - x_q) for x_q in tar_feats])
                for x_p in src_feats
            ]
        )
        covariance_matrix = self.amplitude * torch.exp(
            (-1 / (2 * self.sigma ** 2)) * (distances_array ** 2)
        )

        return covariance_matrix

    def forward(self, src_feats: Tensor, tar_feats: Tensor) -> Tensor:
        return self.get_covariance_matrix(src_feats, tar_feats)
