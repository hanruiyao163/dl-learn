from torch import nn
import torch

from default_boxes import DefaultBoxes


class Loss(nn.Module):
    def __init__(self, dboxes: DefaultBoxes) -> None:
        super().__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy  # 1 / 0.1
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction="none")  # 边界框回归
        self.confidence_loss = nn.CrossEntropyLoss(reduction="none")  # 分类

        # [num_anchors, 4] -> [4, num_anchors] -> [1, num_anchors, 4]
        # 保持和gt维度一致
        self.dboxes = nn.Parameter(
            dboxes(order="xywh").transpose(0, 1).unsqueeze(0), requires_grad=False
        )

    # 论文 计算ground truth相对anchors的回归参数
    def _location_vec(self, loc):
        # 对应gtbox Nx4x8732
        gxy = (
            self.scale_xy
            * (loc[:, :2, :] - self.dboxes[:, :2, :])
            / self.dboxes[:, 2:, :]
        )
        # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    #   预测location 预测标签
    def forward(self, ploc, plabel, gloc, glabel):

        # 获取正样本的mask  Tensor: [N, 8732]
        mask = torch.gt(glabel, 0)

        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # [N, 8732]
        con = self.confidence_loss(plabel, glabel)

        # 获取负样本
        con_neg = con.clone()
        # 正类替换为0
        con_neg[mask] = 0.0
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss
        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(
            pos_num, 0
        ).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(
            dim=0
        )  # 只计算存在正样本的图像损失
        return ret
