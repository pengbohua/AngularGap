from loss_funcs.mmd import MMDLoss
from loss_funcs.adv import LambdaSheduler
import torch
import numpy as np


class LMMDLoss(MMDLoss, LambdaSheduler):
    def __init__(
        self,
        num_class,
        kernel_type="rbf",
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None,
        gamma=1.0,
        max_iter=1000,
        **kwargs
    ):
        """
        Local MMD
        """
        super(LMMDLoss, self).__init__(
            kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs
        )
        super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
        self.num_class = num_class

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == "linear":
            raise NotImplementedError("Linear kernel is not supported yet.")

        elif self.kernel_type == "rbf":
            batch_size = source.size()[0]
            weight_ss, weight_tt, weight_st = self.cal_weight(
                source_label, target_logits
            )
            weight_ss = torch.from_numpy(weight_ss).cuda()  # B, B
            weight_tt = torch.from_numpy(weight_tt).cuda()
            weight_st = torch.from_numpy(weight_st).cuda()

            kernels = self.guassian_kernel(
                source,
                target,
                kernel_mul=self.kernel_mul,
                kernel_num=self.kernel_num,
                fix_sigma=self.fix_sigma,
            )
            loss = torch.Tensor([0]).cuda()
            if torch.sum(torch.isnan(sum(kernels))):
                return loss
            SS = kernels[:batch_size, :batch_size]
            TT = kernels[batch_size:, batch_size:]
            ST = kernels[:batch_size, batch_size:]

            loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
            # Dynamic weighting
            lamb = self.lamb()
            self.step()
            loss = loss * lamb
            return loss, weight_ss.mean(), weight_tt.mean(), weight_st.mean()

    def cal_weight(self, source_label, target_logits):
        batch_size = source_label.size()[0]
        source_label = source_label.cpu().data.numpy()
        source_label_onehot = np.eye(self.num_class)[source_label]  # one hot
        source_label_sum = np.sum(source_label_onehot, axis=0).reshape(
            1, self.num_class
        )
        source_label_sum[source_label_sum == 0] = 100
        source_label_onehot = (
            source_label_onehot / source_label_sum
        )  # one hot src class ratio reciprocal 当前类别样本总数的倒数(类平衡)

        # Pseudo label 目标区域上没有标签只好用伪标签计算类别是否存在67行
        target_label = target_logits.cpu().data.max(1)[1].numpy()

        target_logits = target_logits.cpu().data.numpy()
        target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
        target_logits_sum[target_logits_sum == 0] = 100
        target_logits = (
            target_logits / target_logits_sum
        )  # 不同于src tar这里用的软标签 当前类别样本总数的倒数(类平衡)
        print("target_logits", target_logits)

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(source_label)  #   可能当前batch没有某一类
        set_t = set(target_label)
        count = 0
        for i in range(self.num_class):  #  C)
            if i in set_s and i in set_t:
                s_tvec = source_label_onehot[:, i].reshape(batch_size, -1)  # (B, 1)
                t_tvec = target_logits[:, i].reshape(batch_size, -1)  # (B, 1)

                ss = np.dot(s_tvec, s_tvec.T)  # (B, B)  计算当前batch的加权矩阵
                weight_ss = weight_ss + ss  # 这里为什么累加没看懂
                print(weight_ss)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return (
            weight_ss.astype("float32"),
            weight_tt.astype("float32"),
            weight_st.astype("float32"),
        )


if __name__ == "__main__":
    lmmd = LMMDLoss(4)
    torch.manual_seed(666)
    src_label, tar_logits = torch.LongTensor([0, 3, 0]), torch.softmax(
        torch.randn(3, 4), 1
    )
    print("original logits", tar_logits)
    lmmd.cal_weight(src_label, tar_logits)
