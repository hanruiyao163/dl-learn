import numpy as np


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

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        # 预测层一个cell在原图上的跨度???
        # 8 16 32 64 100 300
        self.steps = steps

        # 每个预测层上default box的scale
        # 21, 45, 99, 153 207 261 315
        self.scales = scales

        # 论文
        fk = fig_size / np.array(steps)

        # 每个预测特征层的 dbox的ratios
        # [2]  [2,3] [2,3] [2,3] [2] [2] 还要再取倒数
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
                for i in range(sfeat):
                    for j in range(sfeat):
                        cx = (j + 0.5) / fk[idx]
                        cy = (i + 0.5) / fk[idx]

                        self.default_boxes.append((cx, cy, w, h))
