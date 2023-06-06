import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform
import paddle
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer, Conv2d, SeparableConvBNReLU, ConvBNReLU

'''
NCD from SINet which built by Deng-Ping Fan in paper "Concealed Object Detection"
'''

__all__ = ['MYNCD', "MYASPP", "MYNECK", "BoundaryBranch"]


@register
class MYNCD(nn.Layer):
    """
        Neighbor Connection Decoder, see https://ieeexplore.ieee.org/document/9444794
        通过MYNCD生成一个c:128的粗略图
        Args:

        """
    __shared__ = ['norm_type']

    def __init__(self, in_channels=256, out_channels=1, norm_decay=0., freeze_norm=False, use_dcn_in_tower=False, norm_type='gn'):
        super(MYNCD, self).__init__()
        self.norm_type = norm_type
        self.use_dcn_in_tower = use_dcn_in_tower
        self.use_dcn = True if self.use_dcn_in_tower else False
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        dim_channels = 32  # 原来论文中的通道数32
        self.conv_dims = nn.LayerList()
        for i in range(3):
            self.conv_dims.append(ConvNormLayer(ch_in=in_channels, ch_out=dim_channels, filter_size=1, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type))

        self.conv_upsample1 = ConvNormLayer(ch_in=dim_channels, ch_out=dim_channels, filter_size=3, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type)
        self.conv_upsample2 = ConvNormLayer(ch_in=dim_channels, ch_out=dim_channels, filter_size=3, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type)
        self.conv_upsample3 = ConvNormLayer(ch_in=dim_channels, ch_out=dim_channels, filter_size=3, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type)
        self.conv_upsample4 = ConvNormLayer(ch_in=dim_channels, ch_out=dim_channels, filter_size=3, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type)
        self.conv_upsample5 = ConvNormLayer(ch_in=dim_channels * 2, ch_out=dim_channels * 2, filter_size=3, stride=1, use_dcn=self.use_dcn, norm_type=self.norm_type)

        self.conv_concat2 = ConvNormLayer(ch_in=dim_channels * 2, ch_out=dim_channels * 2,
                                          filter_size=3, stride=1, norm_type=self.norm_type, norm_decay=self.norm_decay,
                                          freeze_norm=self.freeze_norm,
                                          initializer=XavierUniform(fan_out=in_channels))
        self.conv_concat3 = ConvNormLayer(ch_in=dim_channels * 3, ch_out=dim_channels * 3,
                                          filter_size=3, stride=1, norm_type=self.norm_type, norm_decay=self.norm_decay,
                                          freeze_norm=self.freeze_norm,
                                          initializer=XavierUniform(fan_out=in_channels))
        self.conv4 = ConvNormLayer(ch_in=dim_channels * 3, ch_out=dim_channels,
                                   filter_size=3, stride=1, norm_type=self.norm_type, norm_decay=self.norm_decay,
                                   freeze_norm=self.freeze_norm,
                                   initializer=XavierUniform(fan_out=in_channels))
        self.conv5 = Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        # 空间和通道注意力机制融合
        self.conv_with_c_s_attention = nn.Sequential(
            ChannelAttention(channel=dim_channels),
            SpatialAttetion()
        )

    def forward(self, inputs):
        # 先把通道数从256降到64,通道数改变必要时需要归一化操作。x1,x2,x3,x4分辨率递增。
        x1 = self.conv_dims[0](inputs[3])
        x2 = self.conv_dims[1](inputs[2])
        x3 = self.conv_dims[2](inputs[1])

        x1_1 = x1
        x1_upsample = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x2_1 = self.conv_upsample1(x1_upsample) * x2  # c:32
        x2_1_upsample = F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)
        x2_upsample = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3_1 = self.conv_upsample2(x2_1_upsample) * self.conv_upsample3(x2_upsample) * x3  # c:64

        # myncd struct
        out = paddle.concat((x2_1, x1_upsample), 1)  # c:64
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)  # c:128
        out = paddle.concat((x3_1, out), 1)  # 3* mid_dim

        x = self.conv4(out)  # c:32
        # ncd输出之前，加通道和空间注意力机制
        x = self.conv_with_c_s_attention(x)

        x = self.conv5(x)  # c:1,size:1/8

        return x


@register
@serializable
class MYASPP(nn.Layer):
    """
    对1/16尺寸的特征图进行提取
    """

    def __init__(self, in_channels=1024, out_channels=256):
        super(MYASPP, self).__init__()
        print("--------------using c3 layer from res2net----------------")
        self.aspp = self.add_sublayer("MYASPP_ASPPModule",
                                      ASPPModule(aspp_ratios=[1, 6, 12, 18], in_channels=in_channels, out_channels=out_channels, norm_type='bn'))
        # 空间和通道注意力机制融合
        self.conv_with_c_s_attention = nn.Sequential(
            # self.add_sublayer("MYASPP_ChannelAttention", ChannelAttention(channel=in_channels)),
            self.add_sublayer("MYASPP_SpatialAttetion", SpatialAttetion())
        )

    def forward(self, inputs):
        # input is 1/16
        # x = self.conv_with_c_s_attention(inputs)
        # self.attention_maps = x
        x = self.aspp(inputs)
        x = F.interpolate(x, scale_factor=4, mode='bilinear')  # 1/4,256
        return x


@register
@serializable
class MYNECK(nn.Layer):
    """
    对FPN的特征进行增强
    """
    __shared__ = ['norm_type']

    def __init__(self, in_channels=256, out_channels=256, nums_stage=4, norm_type='bn', norm_decay=0., freeze_norm=False, is_reduce_dim=False, middle_dim=256):
        super(MYNECK, self).__init__()
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.reduce_dim = is_reduce_dim
        self.middle_dim = middle_dim

        self.nums_stage = nums_stage
        self.conv_x = []
        for i in range(self.nums_stage):
            self.conv_x.append(self.add_sublayer("MYNECK_conv_x_{}".format(i),
                                                 ConvNormLayer(ch_in=in_channels * 2 if not self.reduce_dim else out_channels + self.middle_dim,
                                                               ch_out=out_channels,
                                                               filter_size=3,
                                                               stride=1,
                                                               norm_type=self.norm_type,
                                                               norm_decay=self.norm_decay,
                                                               freeze_norm=self.freeze_norm,
                                                               initializer=XavierUniform(fan_out=in_channels))))
        # if no aspp ,using it to expand the channel
        # print("MYNECK: using another conv to input[0]")
        self.expand_conv = self.add_sublayer("MYNECK_expand_conv", ConvNormLayer(ch_in=in_channels,
                                                                                 ch_out=in_channels * 2,
                                                                                 filter_size=3,
                                                                                 stride=1,
                                                                                 norm_type=self.norm_type,
                                                                                 norm_decay=self.norm_decay,
                                                                                 freeze_norm=self.freeze_norm,
                                                                                 initializer=XavierUniform(fan_out=in_channels)))
        if self.reduce_dim:
            print("----------------now reduce the dim to {}----------------".format(self.middle_dim))
            self.third_feat_conv = []
            for i in range(self.nums_stage):
                third_feat_conv_i = self.add_sublayer("MYNECK_third_feat_conv.{}".format(i),
                                                      ConvNormLayer(ch_in=in_channels,
                                                                    ch_out=self.middle_dim,
                                                                    filter_size=3,
                                                                    stride=1,
                                                                    norm_type=self.norm_type,
                                                                    norm_decay=self.norm_decay,
                                                                    freeze_norm=self.freeze_norm,
                                                                    initializer=XavierUniform(fan_out=in_channels)))
                self.third_feat_conv.append(third_feat_conv_i)

    def forward(self, inputs, third_feat=None):
        # input 只选前4层，第五层不要
        out = []
        if third_feat is None:
            x = self.expand_conv(inputs[0])
        else:
            if self.reduce_dim:
                third_feat = self.third_feat_conv[0](third_feat)
            x = paddle.concat([inputs[0], third_feat], axis=1)  # 256*2
        feat = self.conv_x[0](x)
        out.append(feat)
        for i in range(1, self.nums_stage):
            feat = F.interpolate(feat, scale_factor=0.5, mode='bilinear')
            if self.reduce_dim:
                feat = self.third_feat_conv[i](feat)
            x = paddle.concat([inputs[i], feat], axis=1)  # 256*2
            feat = self.conv_x[i](x)
            out.append(feat)
        out.append(F.max_pool2d(out[-1], 1, stride=2))  # 5 stages
        return out


'''
TODO:相关计算放到layers.py中去
'''


class BasicConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, freeze_norm=False,
                 norm_decay=0.):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              weight_attr=ParamAttr(initializer=XavierUniform(fan_out=in_channels)))

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay) if norm_decay is not None else None)
        self.bn = nn.BatchNorm2D(
            out_channels, weight_attr=param_attr, bias_attr=bias_attr)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ChannelAttention(nn.Layer):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias_attr=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        x_maxpool = paddle.reshape(self.max_pool(inputs), (b, c))
        y_avgpool = paddle.reshape(self.avg_pool(inputs), (b, c))
        x_maxpool = self.fc(x_maxpool)
        y_avgpool = self.fc(y_avgpool)
        res = paddle.add(x_maxpool, y_avgpool)
        out = paddle.reshape(self.sigmoid(res), (b, c, 1, 1))
        return out * inputs


class SpatialAttetion(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttetion, self).__init__()
        self.conv1 = Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x_max = paddle.max(inputs, axis=1, keepdim=True)
        y_avg = paddle.mean(inputs, axis=1, keepdim=True)
        # TODO：未来加一层一致性卷积
        out = paddle.concat([x_max, y_avg], axis=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * inputs


# 空间金字塔模块
class ASPPModule(nn.Layer):
    """
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios=[1, 6, 12, 18],
                 in_channels=1024,
                 out_channels=256,
                 align_corners=False,
                 use_sep_conv=False,
                 image_pooling=True,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        super(ASPPModule, self).__init__()
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.LayerList()

        for i, ratio in enumerate(aspp_ratios):
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU

            block = self.add_sublayer('aspp_module.block{}'.format(i), conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format))
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = self.add_sublayer('aspp_module.global_avg_pool', nn.Sequential(
                self.add_sublayer('aspp_module.global_avg_pool.adaptiveAvgPool2D', nn.AdaptiveAvgPool2D(
                    output_size=(1, 1), data_format=data_format)),
                self.add_sublayer('aspp_module.global_avg_pool.conv', ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format))))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = self.add_sublayer('aspp_module.conv_bn_relu.conv', ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format))

        self.dropout = self.add_sublayer('aspp_module.dropout', nn.Dropout(p=0.1))  # drop rate
        # 自己加的
        self.conv_bn = self.add_sublayer('aspp_module.conv_bn.convNormLayer', ConvNormLayer(
            ch_in=out_channels * out_size,
            ch_out=out_channels,
            filter_size=3,
            stride=1,
            norm_type=self.norm_type,
            norm_decay=self.norm_decay,
            freeze_norm=self.freeze_norm,
            initializer=XavierUniform(fan_out=in_channels)))

    def forward(self, x):
        outputs = []
        if self.data_format == 'NCHW':
            interpolate_shape = paddle.shape(x)[2:]
            axis = 1
        else:
            interpolate_shape = paddle.shape(x)[1:3]
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(
                y,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=axis)
        x = self.conv_bn(x)
        # x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


# 空间金字塔模块
class ASPPModule_1_3(nn.Layer):
    """
    因为ASPPModule在最后用的3*3卷积，对目标没有具体的表征。借鉴FPN，所以在这个模块里先用1*1卷积，再3*3卷积。
    Atrous Spatial Pyramid Pooling.

    Args:
        aspp_ratios (tuple): The dilation rate using in ASSP module.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        use_sep_conv (bool, optional): If using separable conv in ASPP module. Default: False.
        image_pooling (bool, optional): If augmented with image-level features. Default: False
    """

    def __init__(self,
                 aspp_ratios=[1, 6, 12, 18],
                 in_channels=1024,
                 out_channels=256,
                 align_corners=False,
                 use_sep_conv=False,
                 image_pooling=True,
                 norm_type='bn',
                 norm_decay=0.,
                 freeze_norm=False,
                 data_format='NCHW'):
        super(ASPPModule_1_3, self).__init__()
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm

        self.align_corners = align_corners
        self.data_format = data_format
        self.aspp_blocks = nn.LayerList()

        for ratio in aspp_ratios:
            if use_sep_conv and ratio > 1:
                conv_func = SeparableConvBNReLU
            else:
                conv_func = ConvBNReLU

            block = conv_func(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 if ratio == 1 else 3,
                dilation=ratio,
                padding=0 if ratio == 1 else ratio,
                data_format=data_format)
            self.aspp_blocks.append(block)

        out_size = len(self.aspp_blocks)

        if image_pooling:
            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2D(
                    output_size=(1, 1), data_format=data_format),
                ConvBNReLU(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias_attr=False,
                    data_format=data_format))
            out_size += 1
        self.image_pooling = image_pooling

        self.conv_bn_relu = ConvBNReLU(
            in_channels=out_channels * out_size,
            out_channels=out_channels,
            kernel_size=1,
            data_format=data_format)

        self.dropout = nn.Dropout(p=0.1)  # drop rate
        # 自己加的
        self.conv_bn = ConvNormLayer(
            ch_in=out_channels * out_size,
            ch_out=out_channels,
            filter_size=1,
            stride=1,
            norm_type=self.norm_type,
            norm_decay=self.norm_decay,
            freeze_norm=self.freeze_norm,
            initializer=XavierUniform(fan_out=out_channels * out_size))
        self.conv_bn_1 = ConvNormLayer(
            ch_in=out_channels,
            ch_out=out_channels,
            filter_size=3,
            stride=1,
            norm_type=self.norm_type,
            norm_decay=self.norm_decay,
            freeze_norm=self.freeze_norm,
            initializer=XavierUniform(fan_out=out_channels))

    def forward(self, x):
        outputs = []
        if self.data_format == 'NCHW':
            interpolate_shape = paddle.shape(x)[2:]
            axis = 1
        else:
            interpolate_shape = paddle.shape(x)[1:3]
            axis = -1
        for block in self.aspp_blocks:
            y = block(x)
            y = F.interpolate(
                y,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(y)

        if self.image_pooling:
            img_avg = self.global_avg_pool(x)
            img_avg = F.interpolate(
                img_avg,
                interpolate_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                data_format=self.data_format)
            outputs.append(img_avg)

        x = paddle.concat(outputs, axis=axis)
        x = self.conv_bn(x)
        x = self.conv_bn_1(x)
        # x = self.conv_bn_relu(x)
        x = self.dropout(x)

        return x


@register
@serializable
class BoundaryBranch(nn.Layer):
    """
    这是一个计算边缘的分支
    Args:
        inputs_feats: [N,256,H,w]
        residual_feat: [N,256,H,w]
    """
    __shared__ = ['norm_type']

    def __init__(self, in_channels=256, out_channels=1, norm_decay=0., freeze_norm=False, use_dcn_in_tower=False, norm_type='gn'):
        super(BoundaryBranch, self).__init__()
        kernel_size = 7
        self.conv1 = self.add_sublayer("spatial_attention", Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False))
        self.edge_conv = self.add_sublayer("edge", Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, inputs_feats, residual_feat):
        x_max = paddle.max(residual_feat, axis=1, keepdim=True)
        x_avg = paddle.mean(residual_feat, axis=1, keepdim=True)
        x = paddle.concat([x_max, x_avg], axis=1)
        x = 1 - F.sigmoid(self.conv1(x))

        out_feats = inputs_feats * x
        x = self.edge_conv(out_feats)  # [N,1,H,W]
        x = F.sigmoid(x)
        return x

# if __name__ == '__main__':
#     x1 = paddle.randn([1, 256, 56, 56])
#     x2 = paddle.randn([1, 256, 112, 112])
#     x3 = paddle.randn([1, 256, 224, 224])
#     x4 = paddle.randn([1, 256, 448, 448])
#     net = NCD()
#     out = net(x1, x2, x3, x4)
#     print(out.shape)
