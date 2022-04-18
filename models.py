import torch.nn as nn
import torch
from angularloss import AngularLoss
from torchvision.models import vgg16_bn, alexnet
from resnet import resnet18, resnet34, resnet50, resnet101
from visualize import ConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"


class Squeeze(nn.Module):
    def forward(self, x):
        if len(x.shape) == 4:
            return x.squeeze(2).squeeze(2)
        elif len(x.shape) == 2:
            return x
        else:
            raise ValueError("invalid input shape")


class Baseline(nn.Module):
    def __init__(self, num_classes=10, latent_dim=512, arch="visualization"):
        super(Baseline, self).__init__()
        self.num_classes = num_classes
        if arch == "visualization":
            self.convlayers = ConvNet(latent_dim=3)
            self.fc_final = nn.Linear(3, num_classes)
        elif arch == "vgg16":
            self.feat = vgg16_bn(pretrained=True, num_classes=latent_dim)
            self.fc_final = nn.Linear(latent_dim, num_classes)
        elif arch == "resnet18":
            self.convlayers = resnet18(pretrained=True, num_classes=latent_dim)
            self.fc_final = nn.Linear(latent_dim, num_classes)
        elif arch == "resnet34":
            self.convlayers = resnet34(pretrained=True, num_classes=latent_dim)
            self.fc_final = nn.Linear(latent_dim, num_classes)
        elif arch == "resnet50":
            self.convlayers = resnet50(pretrained=True, num_classes=latent_dim)
            self.fc_final = nn.Linear(latent_dim, num_classes)
        elif arch == "resnet101":
            self.convlayers = resnet101(pretrained=True, num_classes=latent_dim)
            self.fc_final = nn.Linear(latent_dim, num_classes)

    def forward(self, x, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        x = self.fc_final(x)
        return x


class AngularNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        loss_type="nsl",
        arch="visualization",
        latent_dim=512,
        s=None,
        m=None,
    ):
        super(AngularNet, self).__init__()
        self.num_classes = num_classes
        if arch == "visualization":
            self.feat = ConvNet(latent_dim=3)
            self.feat.out_features = 3
            self.angular_loss = AngularLoss(
                3, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "alexnet":
            self.feat = alexnet(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "vgg16":
            self.feat = vgg16_bn(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "resnet18":
            self.feat = resnet18(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "resnet34":
            self.feat = resnet34(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "resnet50":
            self.feat = resnet50(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        elif arch == "resnet101":
            self.feat = resnet101(pretrained=True)
            self.angular_loss = AngularLoss(
                latent_dim, num_classes, loss_type=loss_type, s=s, m=m
            )
        else:
            raise NotImplementedError
        if arch == "visualization":
            self.linear_project = nn.Identity()
        else:
            self.linear_project = nn.Linear(self.feat.out_features, latent_dim)

    def forward(self, x, labels=None, embed=False):
        x = self.feat(x)
        x = self.linear_project(x)
        if embed:
            return x
        else:
            logits, cos = self.angular_loss(x, labels)
            return logits, cos
