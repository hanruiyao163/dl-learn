import itertools
import numpy as np
import torch
from torch import nn

class DefaultBoxes:
    def __init__(
        self,
        fig_size,
        feat_size,
        steps,
        scales,
        aspect_ratios,
        scale_xy=0.1,
        scale_wh=0.2,
    ) -> None:
        # 原始图像尺寸
        self.fig_size = fig_size

        # 每个预测层feature map尺寸
        # 38 19 10 5 3 1
        self.feat_size = feat_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # 预测层一个cell在原图上的跨度???
        # 8 16 32 64 100 300
        self.steps = steps

        # 每个预测层上default box的scale
        # 21, 45, 99, 153 207 261 315
        self.scales = scales

        # 论文 the size of k-th square feature map  第k个feature map的大小
        fk = fig_size / np.array(steps)

        # 每个预测特征层的dbox的ratios
        # [2]  [2,3] [2,3] [2,3] [2] [2]
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []

        # 遍历特征层 计算dbox
        for idx, sfeat in enumerate(feat_size):
            scale1 = scales[idx] / fig_size  # 转化为相对值
            scale2 = scales[idx + 1] / fig_size
            scale3 = np.sqrt(scale1 * scale2)

            # 加入1比1的比例
            all_sizes = [(scale1, scale1), (scale3, scale3)]

            for alpha in aspect_ratios[idx]:
                w = scale1 * np.sqrt(alpha)
                h = scale1 / np.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            # 当前特征层对应原图default box
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    # +0.5 对应每个cell中心
                    cx = (j + 0.5) / fk[idx]  # x对应列的方向
                    cy = (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float32)
        # 限制在0-1之间
        self.dboxes.clamp_(min=0, max=1)

        # 将(cx,cy,w,h)转化为左上右下形式
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        elif order == "xywh":
            return self.dboxes
        else:
            raise ValueError("order should be ltrb or xywh")

    @property
    def scale_xy(self):
        return self.scale_xy_
    
    @property
    def scale_wh(self):
        return self.scale_wh_


def dboxes300_coco():
    figsize = 300  # 输入网络的图像大小
    feat_size = [38, 19, 10, 5, 3, 1]  # 每个预测层的feature map尺寸
    steps = [8, 16, 32, 64, 100, 300]  # 每个特征层上的一个cell在原图上的跨度
    scales = [21, 45, 99, 153, 207, 261, 315]  # 每个特征层上预测的default box的scale
    aspect_ratios = [
        [2],
        [2, 3],
        [2, 3],
        [2, 3],
        [2],
        [2],
    ]  # 每个预测特征层上预测的default box的ratios
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


if __name__ == "__main__":
    dboxes = dboxes300_coco()
    print(dboxes.dboxes.size())
    print(dboxes.dboxes_ltrb.size())
    print(dboxes.dboxes[:5])
    print(dboxes.dboxes_ltrb[:5])
