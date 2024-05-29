# 使用resnet50代替vgg16
# 移除conv4_x之后的所有层
# conv4_x所有层的stride=1
# bias移除weight decay
# 检测层加入batch normalization


import d2l
import torch.nn as nn

from resnet50_backbone import resnet50


class Backbone(nn.Module):
    def __init__(self, pretrain_path=None):
        super().__init__()
        net = resnet50()
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        if pretrain_path is not None:
            net.load_state_dict(d2l.load(pretrain_path))

        # 只到layer3 layer4之后的不要
        self.feature_extractor = nn.Sequential(*list(net.children())[:-2])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD300(nn.Module):
    def __init__(self, backbone=None, num_classes=21) -> None:
        super().__init__()
        if backbone is None:
            raise Exception("backbone is None")
        self.featuer_extractor = backbone

        self.num_classes = num_classes
        self._build_additional_features(self.featuer_extractor.out_channels)

        self.num_defaults = [4, 6, 6, 6, 4, 4]
        location_extractors = []
        confidence_extractors = []

        # out_channels = [1024, 512, 512, 256, 256, 256]
        for num_dboxes, outc in zip(
            self.num_defaults, self.featuer_extractor.out_channels
        ):
            location_extractors.append(
                nn.Conv2d(outc, num_dboxes * 4, kernel_size=3, padding=1)
            )
            confidence_extractors.append(
                nn.Conv2d(outc, num_dboxes * self.num_classes, kernel_size=3, padding=1)
            )

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        default_boxes = dboxes300_coco()
        self.compute_loss = Loss(default_boxes)
        self.encoder = Encoder(default_boxes)
        self.postprocess = PostProcess(default_boxes)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # 额外的特征提取器
    def _build_aditional_features(self, in_channels):
        additional_blocks = []
        middle_channels = [256, 256, 128, 128, 128]
        for i, (inc, outc, midc) in enumerate(
            zip(in_channels[:-1], in_channels[1:], middle_channels)
        ):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(inc, midc, kernel_size=1, bias=False),
                nn.BatchNorm2d(midc),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    midc,
                    outc,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(outc),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(additional_blocks)

    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = d2l.cat(locs, 2).contiguous(), d2l.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        x = self.featuer_extractor(image)
        # 👆38x38x1024 👇19x19x512 10x10x512 5x5x256 3x3x256 1x1x256

        # detection_features = torch.jit.annotate(List[torch.Tensor], [])
        detection_features: list[d2l.Tensor] = []
        detection_features.append(x)
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)

        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # nbatch x 8732 x {nlabels, nlocs}


if __name__ == "__main__":
    a = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
    )
    b = nn.Sequential(
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
    )
    model = nn.ModuleList([a, b])
    print(model[0])
