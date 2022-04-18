import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularLoss(nn.Module):
    def __init__(
        self, in_features, out_features, loss_type="arcface", eps=1e-7, s=None, m=None
    ):
        """
        AngularLoss
        Four 'loss_types' available: ['softmax loss', 'normalized softmax loss', 'arcface loss', 'cosface']
        ArcFace: https://arxiv.org/abs/1801.07698
        CosFace: https://arxiv.org/abs/1801.05599
        """
        super(AngularLoss, self).__init__()
        self.num_classes = out_features
        loss_type = loss_type.lower()
        assert loss_type in ["sl", "nsl", "arcface", "cosface"]
        if loss_type == "arcface":
            self.s = torch.ones(1) * 64.0 if not s else torch.ones(1) * s
            self.m = torch.ones(1, device="cuda") * 0.5 if not m else torch.ones(1) * m
        if loss_type == "cosface":
            self.s = torch.ones(1) * 30.0 if not s else torch.ones(1) * s
            self.m = torch.ones(1) * 0.35 if not m else torch.ones(1) * m
        if loss_type == "nsl":
            self.s = torch.ones(1) * 15.0 if not s else torch.ones(1) * s
            self.m = torch.ones(1) * 0.0 if not m else torch.ones(1) * m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.fc_softmax = nn.Linear(in_features, out_features, bias=True)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """

        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        if self.loss_type == "sl":
            logits = self.fc_softmax(x)
            W = F.normalize(self.fc_softmax.weight.data, p=2, dim=1)  # C x F
            x = F.normalize(x, p=2, dim=1)  # B x F
            cosine = torch.mm(x, W.t().contiguous())  # B x C
            return logits, cosine.detach()

        W = F.normalize(self.fc.weight.data, p=2, dim=1)  # C x F

        x = F.normalize(x, p=2, dim=1)  # B x F
        # cosine dists
        cosine = torch.mm(x, W.t().contiguous())  # B x C

        # move s, m to the same device as wf
        self.s = self.s.to(x.device)
        self.m = self.m.to(x.device)

        if self.loss_type == "nsl":
            logits = cosine * self.s
        elif self.loss_type == "cosface":
            m_hot = nn.functional.one_hot(labels, num_classes=self.num_classes) * self.m
            cosine_m = cosine - m_hot
            logits = cosine_m * self.s
        elif self.loss_type == "arcface":
            m_hot = nn.functional.one_hot(labels, num_classes=self.num_classes) * self.m
            cosine_m = cosine.clamp(-1.0 + self.eps, 1 - self.eps).acos()
            cosine_m += m_hot
            logits = cosine_m.cos() * self.s
        else:
            raise NotImplementedError
        return logits, cosine.detach()
