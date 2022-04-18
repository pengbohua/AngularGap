import torch.nn as nn
import torch
import torch.nn.functional as F

class VGGPD(nn.Module):
    def __init__(
            self,
            encoder=None,
            num_classes=100
    ):
        super(VGGPD, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, x, k=0, train=True):
        """

        :param x:
        :param k: output fms from the kth conv2d or the last layer
        :return:
        """
        n_layer = 0
        _fm = None
        for m in self.encoder.children():
            x = m(x)
            if not train:
                if isinstance(m, nn.Conv2d):
                    if n_layer == k:
                        return None, x.view(x.shape[0], -1) # B x (C x F x F)
                    n_layer += 1
        logits = self.classifier(x)
        if not train:
            if k == n_layer:
                _fm = torch.softmax(logits, 1)
                return None, _fm.view(_fm.shape[0], -1)  # B x (C x F x F)
        else:
            return logits


class MLP7(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP7, self).__init__()
        test_in = torch.randn(1, 3, 32, 32).view(1, -1)
        self.fl = nn.Flatten()
        self.d1 = nn.Linear(test_in.shape[1], 2048)
        self.d2 = nn.Linear(2048, 2048)
        self.d3 = nn.Linear(2048, 2048)
        self.d4 = nn.Linear(2048, 2048)
        self.d5 = nn.Linear(2048, 2048)
        self.d6 = nn.Linear(2048, 2048)
        self.d7 = nn.Linear(2048, num_classes)

    def forward(self, x, k=0, train=True):
        representations = []
        f1 = self.d1(self.fl(x))
        representations.append(f1) # B x 1 x F
        f2 = self.d2(torch.relu_(f1))
        representations.append(f2)
        f3 = self.d3(torch.relu_(f2))
        representations.append(f3)
        f4 = self.d4(torch.relu_(f3))
        representations.append(f4)
        f5 = self.d5(torch.relu_(f4))
        representations.append(f5)
        f6 = self.d6(torch.relu_(f5))
        representations.append(f6)
        logits = self.d7(torch.relu_(f6))

        # the last representation is added after softmax
        f7 = torch.softmax(logits, dim=1)
        representations.append(f7)
        if train:
            return logits
        else:
            return None, representations[k]

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t, rm_top1=True, dist='l2'):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    B, F = feature.shape
    K, F = feature_bank.shape
    if dist =='cosine':
        feature = F.normalize(feature, dim=1, p=2.0)
        feature_bank = F.normalize(feature_bank, dim=1, p=2.0)     # normalize feature dim
        feature.mul_(feature_bank.t().contiguous()) # similarity
    elif dist =='l2':
        feature = feature.unsqueeze(1).expand(B, K, F)
        feature_bank = feature_bank.unsqueeze(0).expand(B, K, F)
        feature.sub_(feature_bank).pow_(2).sum_(2)  # similarity
    else:
        raise NotImplementedError

    # [B, K]
    if rm_top1:
        sim_weight_add_one, sim_indices_add_one = feature.topk(k=(knn_k + 1), dim=-1)
        sim_weight, sim_indices = sim_weight_add_one[:, 1:], sim_indices_add_one[:, 1:]   # remove the nearest pt of current evaluating pt in the train split
    else:
        sim_weight, sim_indices = feature.topk(k=knn_k, dim=-1)
    # [B, K] labels for all pts in feature bank along dim1
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # pred_prob = F.normalize(pred_scores, p=1, dim=1)
    # pred_labels = pred_scores.argsort(dim=-1, descending=True)      # rank the knn labels
    return pred_scores

class BasicBlockPD(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockPD, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, train=True):
        out = F.relu_(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if not train:
            return None, out
        else:
            out = F.relu(out)
            return out


class ResNetPD(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(ResNetPD, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, k=0, train=True):
        '''

        :param x:
        :param k:
        :param train: switch model to test and extract the FMs of the kth layer
        :return:
        '''
        i = 0
        out = self.bn1(self.conv1(x))
        if k==i and not(train):
            return None, out.view(out.shape[0], -1)
        out = torch.relu_(out)
        i +=1
        for module in self.layer1:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1

        for module in self.layer2:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        for module in self.layer3:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        for module in self.layer4:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        if k == i and not (train):
            _f = F.softmax(out, 1)  # take the output of softmax
            return None, _f
        else:
            return out

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BasicBlockWS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockWS, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, self.expansion*planes)
            )

    def forward(self, x, train=True):
        out = F.relu_(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        if not train:
            return None, out
        else:
            out = F.relu(out)
            return out


class ResNetWS(nn.Module):
    '''
    We use Conv2d (weight standardization) to replace nn.Conv2d and Group norm to replace BN2d
    '''
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(ResNetWS, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, k=0, train=True):
        '''

        :param x:
        :param k:
        :param train: switch model to test and extract the FMs of the kth layer
        :return:
        '''
        i = 0
        out = self.gn1(self.conv1(x))
        if k==i and not(train):
            return None, out.view(out.shape[0], -1)
        out = torch.relu_(out)
        i +=1
        for module in self.layer1:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1

        for module in self.layer2:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        for module in self.layer3:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        for module in self.layer4:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = torch.relu_(out)
            i+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        if k == i and not (train):
            _f = F.softmax(out, 1)  # take the output of softmax
            return None, _f
        else:
            return out
