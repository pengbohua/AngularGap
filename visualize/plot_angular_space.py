import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def plot2d(embeds, labels, num_classes, fig_path="./unit2d.pdf"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    xlabels = [
        "airplane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "fog",
        "horse",
        "ship",
        "truck",
    ]

    embeds = F.normalize(embeds, dim=1)
    embeds = embeds.cpu().numpy()
    for i in range(num_classes):
        ax.scatter(
            embeds[labels == i, 0], embeds[labels == i, 1], label=xlabels[i], s=10
        )

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.set_aspect(1)
    plt.subplots_adjust(right=0.75)
    plt.savefig(fig_path)
    plt.show()


def plot3d(embeds, labels, num_classes, fig_path="./unit3d.pdf"):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    # theta = np.zeros_like(theta)
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="w", alpha=0.3, linewidth=0)
    embeds = F.normalize(embeds, dim=1)
    embeds = embeds.cpu().numpy()
    xlabels = [
        "airplane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "fog",
        "horse",
        "ship",
        "truck",
    ]
    for i in range(num_classes):
        ax.scatter(
            embeds[labels == i, 0],
            embeds[labels == i, 1],
            embeds[labels == i, 1],
            label=xlabels[i],
            s=10,
        )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.show()


class ConvNet(nn.Module):
    def __init__(self, latent_dim):
        super(ConvNet, self).__init__()
        self.latent_dim = latent_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=8, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_projection = nn.Linear(512, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(batch_size, -1)
        x = self.fc_projection(x)
        return x
