# TEM module
# neck which contains TEMS
# solo head module
# trans solo mask head module
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform, Normal, KaimingNormal

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer, ConvBNReLU, DeformableConvV2, Conv2d
from .. import FPN
from ..shape_spec import ShapeSpec

__all__ = ['TEM', 'TransMaskHead', 'FPN_BS']


@register
@serializable
class TEM(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channel=32,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 cascade_attentions=True,
                 boundary_learn=True,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 relu_before_extra_convs=True):
        super(TEM, self).__init__()
        print("TEM:\n\tcascade_attentions: {}".format(cascade_attentions))  # 打印信息
        print("\tboundary_learn: {}".format(boundary_learn))  # 打印信息
        self.boundary_learn = boundary_learn
        self.cascade_attentions = cascade_attentions
        self.spatial_scales = spatial_scales
        self.out_channel = out_channel
        self.tem_conv = []

        # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
        # 0 <= st_stage < ed_stage <= 3
        st_stage = 4 - len(in_channels)  # 选择res_x进行计算
        ed_stage = st_stage + len(in_channels) - 1
        for i in range(st_stage, ed_stage + 1):
            lateral_name = 'tem_layer_res{}'.format(i + 2)
            in_c = in_channels[i - st_stage]
            lateral = nn.Sequential()
            lateral.add_sublayer(lateral_name, TemBlock(in_channel=in_c, out_channel=out_channel, tem_ratios=[1, 3, 5, 7]))
            self.add_sublayer('res{}_conv_seq'.format(i + 2), lateral)
            self.tem_conv.append(lateral)

        self.boundary_conv = Boundary_Attention(in_channel=out_channel, out_channel=1)

        tem_conv_c3 = nn.Sequential()
        # tem_conv_c3.add_sublayer('tem_conv_c3_seq.conv1',
        #                          ConvNormLayer(ch_in=out_channel,
        #                                        ch_out=out_channel,
        #                                        filter_size=3,
        #                                        stride=1,
        #                                        padding=1,
        #                                        initializer=XavierUniform(fan_out=out_channel))
        #                          )
        # tem_conv_c3.add_sublayer('tem_conv_c3_seq.act1', nn.GELU())
        tem_conv_c3.add_sublayer('tem_conv_c3_seq.conv2',
                                 ConvNormLayer(ch_in=in_channels[0],
                                               ch_out=out_channel,
                                               filter_size=1,
                                               stride=1,
                                               initializer=XavierUniform(fan_out=out_channel))
                                 )
        tem_conv_c3.add_sublayer('tem_conv_c3_seq.act2', nn.GELU())
        self.add_sublayer('tem_conv_c3_seq', tem_conv_c3)
        self.tem_conv_c3 = tem_conv_c3

        tem_conv_c5 = nn.Sequential()
        tem_conv_c5.add_sublayer('tem_layer.tem_conv_c5.conv1',
                                 Conv2d(in_channels=out_channel * 2,
                                        out_channels=out_channel,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                        weight_init=KaimingNormal())
                                 )
        # tem_conv_c5.add_sublayer('tem_layer.tem_conv_c5.act1', nn.GELU())
        self.add_sublayer('tem_conv_c5_seq', tem_conv_c5)
        self.tem_conv_c5 = tem_conv_c5

        self.cascade_attention_module = CascadeAttentions(in_channel=in_channels[0], out_channel=in_channels[0])

        preprocess_body_feat3 = nn.Sequential()
        preprocess_body_feat3_name = 'tem_layer.preprocess_body_feat1'
        preprocess_body_feat3.add_sublayer(preprocess_body_feat3_name + '.conv.1',
                                           ConvNormLayer(ch_in=in_channels[0],
                                                         ch_out=in_channels[0],
                                                         filter_size=1,
                                                         stride=1,
                                                         initializer=XavierUniform(fan_out=in_channels[0]))
                                           )
        preprocess_body_feat3.add_sublayer(preprocess_body_feat3_name + '.conv.1.act', nn.GELU())
        preprocess_body_feat3.add_sublayer(preprocess_body_feat3_name + '.conv.2',
                                           ConvNormLayer(ch_in=in_channels[0],
                                                         ch_out=in_channels[0],
                                                         filter_size=1,
                                                         stride=1,
                                                         initializer=XavierUniform(fan_out=in_channels[0]))
                                           )
        preprocess_body_feat3.add_sublayer(preprocess_body_feat3_name + '.conv.2.act', nn.GELU())
        self.add_sublayer(preprocess_body_feat3_name, preprocess_body_feat3)
        self.preprocess_body_feat3 = preprocess_body_feat3

        preprocess_body_feat_5 = nn.Sequential()
        preprocess_body_feat_5_name = 'tem_layer.preprocess_body_feat_5'
        preprocess_body_feat_5.add_sublayer(preprocess_body_feat_5_name + '.conv1',
                                            Conv2d(in_channels=out_channel,
                                                   out_channels=out_channel,
                                                   kernel_size=1,
                                                   stride=1,
                                                   bias=False,
                                                   weight_init=KaimingNormal())
                                            )
        self.add_sublayer(preprocess_body_feat_5_name, preprocess_body_feat_5)
        self.preprocess_body_feat_5 = preprocess_body_feat_5

        self.postprocess_t5 = self.add_sublayer('tem_layer.postprocess_t5.conv',
                                                Conv2d(in_channels=out_channel * 2,
                                                       out_channels=out_channel,
                                                       kernel_size=1,
                                                       stride=1,
                                                       bias=False,
                                                       weight_init=KaimingNormal())
                                                )
        self.preprocess_body_feat4 = self.add_sublayer('tem_layer.preprocess_body_feat_4.conv',
                                                       ConvBNReLU(in_channels=in_channels[1],
                                                                  out_channels=in_channels[0],
                                                                  kernel_size=1,
                                                                  stride=1,
                                                                  bias_attr=False)
                                                       )

    def forward(self, body_feats):
        # 通道数为64
        body_feat_c3 = body_feats[0]
        tem_out = []
        num_levels = len(body_feats)
        for i in range(num_levels - 1, -1, -1):
            if i == 0:
                preprocess_feats = self.preprocess_body_feat3(body_feats[i])  # 2个1*1CBG
                # TODO:用C4赋值C3阶段语义
                preprocess_feats_c4 = F.interpolate(body_feats[i + 1], scale_factor=2.0, mode='bilinear')  # size=preprocess_feats.shape[-2:]
                preprocess_feats_c4 = self.preprocess_body_feat4(preprocess_feats_c4)  # C->512
                body_feats[i] = paddle.add(preprocess_feats, preprocess_feats_c4 * 5)  # 增强5倍
                # TODO:在对C3增强语义后,添加cascade_attention_module
                if self.cascade_attentions:
                    body_feats[i] = self.cascade_attention_module(body_feats[i])  # C:512->1024->512
            tem_out.append(self.tem_conv[i](body_feats[i]))  # 级联式对不同通道数的stage进行空洞卷积，在C4阶段多加2个卷积层
        tem_out = tem_out[::-1]

        # TODO:T1自增强5倍
        tem_out[1] = tem_out[1] * 5
        # TODO:用C4增强C5的语义
        tem_out[2] = self.preprocess_body_feat_5(tem_out[2])
        preprocess_feats = F.interpolate(F.gelu(tem_out[1]), scale_factor=0.5, mode='bilinear')
        tem_out[2] = paddle.concat([preprocess_feats, tem_out[2]], axis=1)
        tem_out[2] = self.tem_conv_c5(tem_out[2])  # 数值降低

        # TODO:用C3增强C5的边缘,不可激活
        preprocess_feats = self.tem_conv_c3(body_feat_c3)  # 出现语义
        preprocess_feats = F.interpolate(preprocess_feats, scale_factor=0.25, mode='bilinear')
        tem_out[2] = paddle.concat([preprocess_feats, tem_out[2]], axis=1)  # 边缘不必增强，不然会扰乱最大值
        tem_out[2] = self.postprocess_t5(tem_out[2])  # tem_out[2]的最终结果

        # TODO:用并联式注意力机制关注重要特征
        # if self.cascade_attentions:
        #     tem_out[0] = self.cascade_attention_module(tem_out[0])  # C:64->1024->64

        # TODO：边缘监督学习
        boundary_map = []
        if self.boundary_learn:
            boundary_map = self.boundary_conv(tem_out, body_feats[0])  # C:1

        return tem_out, boundary_map

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]


@register
class TemBlock(nn.Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 branch_nums=4,
                 tem_ratios=[1, 3, 5, 7],
                 norm_type=None,
                 data_format='NCHW'):
        super(TemBlock, self).__init__()
        print("TemBlock:\t\ntem_ratios: {}".format(tem_ratios))  # 打印信息
        self.branch_nums = branch_nums
        self.branches_conv = []
        for i in range(branch_nums):
            branch_name = 'tem_block_branch{}'.format(i)
            conv_branch_feat = nn.Sequential()
            if i == 0:
                conv_branch_feat.add_sublayer(branch_name + '.conv1',
                                              ConvNormLayer(ch_in=in_channel,
                                                            ch_out=out_channel,
                                                            filter_size=1,
                                                            stride=1,
                                                            initializer=XavierUniform(fan_out=in_channel))
                                              )
                conv_branch_feat.add_sublayer(branch_name + '.act1', nn.GELU())
                self.add_sublayer(branch_name, conv_branch_feat)
                self.branches_conv.append(conv_branch_feat)
                continue
            conv_branch_feat.add_sublayer(branch_name + '.conv1',
                                          ConvNormLayer(ch_in=in_channel,
                                                        ch_out=out_channel,
                                                        filter_size=1,
                                                        stride=1,
                                                        initializer=XavierUniform(fan_out=in_channel)))
            conv_branch_feat.add_sublayer(branch_name + '.conv1.act', nn.GELU())
            # conv_branch_feat.add_sublayer(branch_name + '.conv2',
            #                               ConvNormLayer(ch_in=out_channel,
            #                                             ch_out=out_channel,
            #                                             filter_size=(1, tem_ratios[i]),
            #                                             padding=(0, (tem_ratios[i] - 1) // 2),
            #                                             stride=1,
            #                                             initializer=XavierUniform(fan_out=in_channel)))
            # conv_branch_feat.add_sublayer(branch_name + '.conv3',
            #                               ConvNormLayer(ch_in=out_channel,
            #                                             ch_out=out_channel,
            #                                             filter_size=(tem_ratios[i], 1),
            #                                             padding=((tem_ratios[i] - 1) // 2, 0),
            #                                             stride=1,
            #                                             initializer=XavierUniform(fan_out=in_channel)))
            conv_branch_feat.add_sublayer(branch_name + '.conv4',
                                          ConvBNReLU(in_channels=out_channel,
                                                     out_channels=out_channel,
                                                     kernel_size=3,
                                                     dilation=tem_ratios[i],
                                                     padding=0 if tem_ratios[i] == 1 else tem_ratios[i],
                                                     bias_attr=False)
                                          )
            self.add_sublayer(branch_name, conv_branch_feat)
            self.branches_conv.append(conv_branch_feat)
        self.conv_cat = self.add_sublayer("tem_block_conv_cat",
                                          ConvBNReLU(in_channels=out_channel * (self.branch_nums + 2),
                                                     out_channels=out_channel,
                                                     kernel_size=1,
                                                     stride=1,
                                                     bias_attr=False)
                                          )
        self.conv_res = self.add_sublayer('tem_block_conv_res',
                                          Conv2d(in_channels=in_channel,
                                                 out_channels=out_channel,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=False,
                                                 weight_init=KaimingNormal())
                                          )
        # self.relu = self.add_sublayer("tem_block_act_res", nn.ReLU())
        # 自己加的
        # self.conv_out = self.add_sublayer('tem_block.conv_out',
        #                                   ConvNormLayer(ch_in=out_channel,
        #                                                 ch_out=out_channel,
        #                                                 filter_size=3,
        #                                                 stride=1,
        #                                                 norm_type='bn',
        #                                                 initializer=XavierUniform(fan_out=in_channel))
        #                                   )
        self.conv_out = self.add_sublayer('tem_block.conv_out',
                                          Conv2d(in_channels=out_channel,
                                                 out_channels=out_channel,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=False,
                                                 weight_init=KaimingNormal())
                                          )

        attention_branch_name = 'tem_block_attention'
        attention_branch_feat = nn.Sequential()
        attention_branch_feat.add_sublayer(attention_branch_name + '.conv1',
                                           ConvNormLayer(ch_in=in_channel,
                                                         ch_out=out_channel,
                                                         filter_size=3,
                                                         stride=1,
                                                         norm_type='bn',
                                                         initializer=XavierUniform(fan_out=in_channel))
                                           )
        # attention_branch_feat.add_sublayer(attention_branch_name + '.act1', nn.GELU())
        attention_branch_feat.add_sublayer(attention_branch_name + '.conv2',
                                           Conv2d(in_channels=out_channel,
                                                  out_channels=out_channel,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=False,
                                                  weight_init=KaimingNormal())
                                           )
        attention_branch_feat.add_sublayer(attention_branch_name + '.conv3', SpatialAttetion())
        self.add_sublayer(attention_branch_name, attention_branch_feat)
        self.attention_conv = attention_branch_feat

    def forward(self, stage_feat):
        block_out = []  # [N,64*5,H,W]
        for i in range(self.branch_nums):
            block_out.append(self.branches_conv[i](stage_feat))
            # TODO:如果形状不对齐，则用interpolate
        block_out.append(self.attention_conv(stage_feat))
        # x0 = self.branch0(stage_feat)
        # x1 = self.branch1(stage_feat)
        # x2 = self.branch2(stage_feat)
        # x3 = self.branch3(stage_feat)
        x_cat = self.conv_cat(paddle.concat(block_out + [self.conv_res(stage_feat)], axis=1))
        # x = paddle.add(x_cat, self.conv_res(stage_feat))
        x = self.conv_out(x_cat)  # 替换self.relu
        return x


@register
class Boundary_Attention(nn.Layer):
    def __init__(self,
                 in_channel,
                 out_channel,
                 use_deformable=False,
                 data_format='NCHW'):
        super(Boundary_Attention, self).__init__()
        self.use_deformable = use_deformable

        self.conv_t4 = self.add_sublayer('boundary_conv_t4',
                                         ConvNormLayer(ch_in=in_channel,
                                                       ch_out=in_channel,
                                                       filter_size=3,
                                                       stride=1,
                                                       # padding=1,
                                                       initializer=XavierUniform(fan_out=in_channel))
                                         )

        self.conv_t3 = self.add_sublayer("boundary_conv_t3",
                                         ConvNormLayer(ch_in=in_channel + 512,
                                                       ch_out=in_channel,
                                                       filter_size=3,
                                                       stride=1,
                                                       # padding=1,
                                                       initializer=XavierUniform(fan_out=in_channel)))
        self.conv1 = self.add_sublayer("boundary_conv1",
                                       ConvNormLayer(ch_in=in_channel,
                                                     ch_out=in_channel,
                                                     filter_size=3,
                                                     stride=1,
                                                     # padding=1,
                                                     initializer=XavierUniform(fan_out=in_channel))
                                       )
        if self.use_deformable:
            # TODO:需要加BN
            self.conv2 = self.add_sublayer("boundary_conv2",
                                           DeformableConvV2(
                                               in_channels=in_channel,
                                               out_channels=out_channel,
                                               kernel_size=1,
                                               weight_attr=ParamAttr(initializer=Normal(0, 0.01))))
        else:
            self.conv2 = self.add_sublayer("boundary_conv2",
                                           Conv2d(in_channels=in_channel,
                                                  out_channels=out_channel,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=1,
                                                  bias=False,
                                                  weight_init=KaimingNormal()))
        self.conv3 = self.add_sublayer("boundary_conv3",
                                       Conv2d(
                                           in_channels=out_channel,
                                           out_channels=out_channel,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False,
                                           weight_init=KaimingNormal())
                                       )

        attention_weight_conv_seq = nn.Sequential()
        attention_weight_conv_seq.add_sublayer('boundary_attention_weight_conv_seq.conv',
                                               Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=7 // 2, bias=False, weight_init=KaimingNormal()))
        attention_weight_conv_seq.add_sublayer('boundary_attention_weight_conv_seq.sigmoid', nn.Sigmoid())
        self.add_sublayer('boundary_attention_weight_conv_seq', attention_weight_conv_seq)
        self.attention_weight_conv = attention_weight_conv_seq

    def forward(self, tem_feats, backbone_feat):
        # 利用tem_feat的几何信息,生成边缘. bockbone_feat: 512
        # x = t3 concat backbone_feat->conv+bn+gelu
        x = paddle.concat([tem_feats[0], backbone_feat], axis=1)
        x = F.gelu(self.conv_t3(x))
        # x = t4 add x ->conv+bn+gelu
        feat = F.interpolate(tem_feats[1], scale_factor=2.0, mode='bilinear', align_corners=True)
        x = paddle.add(feat, x)
        x = F.gelu(self.conv_t4(x))
        # x = t5 * t -> conv+bn
        feat = F.interpolate(tem_feats[2], scale_factor=4.0, mode='bilinear', align_corners=True)
        x = paddle.multiply(feat, x)
        x = self.conv1(x)

        # TODO:做注意力机制，用得到的权重乘x
        avg_x = paddle.mean(x, axis=1, keepdim=True)
        max_x = paddle.max(x, axis=1, keepdim=True)
        attention_weight_x = paddle.concat([avg_x, max_x], 1)
        attention_weight_x = self.attention_weight_conv(attention_weight_x)  # include sigmoid

        # x = attention_weight_x * x
        # TODO:利用前景和背景（0-1）的异或运算生成边缘
        # fg_weight = paddle.where(attention_weight_x >= 0.99, paddle.to_tensor(1.), paddle.to_tensor(0.))  # 前景权重
        # bg_weight = paddle.where(attention_weight_x >= 0.7, paddle.to_tensor(1.), paddle.to_tensor(0.))
        # boundary_weight = paddle.logical_xor(bg_weight, fg_weight)
        fg_weight = paddle.where(attention_weight_x > 0.5, attention_weight_x, paddle.to_tensor(0.1))
        boundary_weight = paddle.cast(fg_weight, dtype='float32')

        # 替换
        # x = attention_weight_x * backbone_feat
        x = boundary_weight * F.gelu(x)
        x = self.conv2(x)  # C:1
        x = self.conv3(x)  # C:1
        return x


@register
class CascadeAttentions(nn.Layer):
    def __init__(self,
                 in_channel=512,
                 mid_channel=1024,
                 out_channel=512):
        super(CascadeAttentions, self).__init__()
        print("CascadeAttentions:\n\tmid_channel: {}".format(mid_channel))  # 打印信息
        self.mid_channel = mid_channel
        self.channel_attetnion = ChannelAttention(channel=self.mid_channel)
        self.spatial_attetnion = SpatialAttetion()
        self.conv = self.add_sublayer('cascade_attentions.conv',
                                      ConvNormLayer(ch_in=self.mid_channel,
                                                    ch_out=in_channel,
                                                    filter_size=1,
                                                    stride=1,
                                                    initializer=XavierUniform(fan_out=in_channel))
                                      )
        self.conv_feat = self.add_sublayer('cascade_attentions.conv_feat',
                                           Conv2d(in_channels=in_channel,
                                                  out_channels=in_channel,
                                                  kernel_size=1,
                                                  stride=1,
                                                  bias=False,
                                                  weight_init=KaimingNormal())
                                           )

        self.conv_input_feat = self.add_sublayer('cascade_attentions.conv_input_feat',
                                                 ConvBNReLU(in_channels=in_channel,
                                                            out_channels=self.mid_channel,
                                                            kernel_size=1,
                                                            stride=1,
                                                            bias_attr=False)
                                                 )
        self.spatial_attetnion_feat = self.add_sublayer('cascade_attentions.spatial_attetnion_feat',
                                                        ConvNormLayer(ch_in=in_channel,
                                                                      ch_out=self.mid_channel,
                                                                      filter_size=1,
                                                                      stride=1,
                                                                      initializer=XavierUniform(fan_out=in_channel))
                                                        )

    def forward(self, input_feat):
        # T3:64
        # x = input_feat
        x = self.conv_input_feat(input_feat)  # C:1024
        x1 = self.channel_attetnion(x)  # C:1024
        x2 = self.spatial_attetnion_feat(input_feat)  # C:1024
        x2 = self.spatial_attetnion(x2)  # C:1024
        # x = paddle.concat([x1, x2], axis=1)  # C:1024*2
        x = paddle.add(x1, x2)  # C:1024
        x = F.gelu(self.conv(x))  # C:64

        x = paddle.add(x, input_feat)
        x = self.conv_feat(x)
        # x = F.gelu(x)  # C:64
        return x


@register
class TransMaskHead(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels=64,
                 mid_channels=128,
                 out_channels=256,
                 start_level=0,
                 end_level=3,
                 use_dcn_in_tower=False,
                 use_same_kernels=True,
                 norm_type='gn',
                 boundary_learn=True):
        super(TransMaskHead, self).__init__()
        print("TransMaskHead:\n\tboundary_learn: {}".format(boundary_learn))  # 打印信息
        self.boundary_learn = boundary_learn
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.range_level = end_level - start_level
        self.convs_all_levels = []
        self.norm_type = norm_type
        self.use_dcn_in_tower = use_dcn_in_tower
        self.use_same_kernels = use_same_kernels
        self.use_dcn = True if self.use_dcn_in_tower else False
        for i in range(start_level, end_level + 1):
            conv_feat_name = 'MaskHead.convs_all_levels.{}'.format(i)
            conv_pre_feat = nn.Sequential()
            if i == end_level:
                conv_pre_feat.add_sublayer(conv_feat_name + '.{}_concat_boundary'.format(i),
                                           ConvNormLayer(
                                               ch_in=self.mid_channels,
                                               ch_out=self.mid_channels,
                                               filter_size=3,
                                               stride=1)
                                           )
                conv_pre_feat.add_sublayer(
                    conv_feat_name + '.{}_act1'.format(i), nn.GELU())
                # 上采样，再卷积
                # conv_pre_feat.add_sublayer(conv_feat_name + '.{}_upsample'.format(i),
                #                            nn.Upsample(
                #                                scale_factor=2, mode='bilinear'))
                # conv_pre_feat.add_sublayer(conv_feat_name + '.{}_conv'.format(i),
                #                            ConvNormLayer(
                #                                ch_in=self.in_channels,
                #                                ch_out=self.mid_channels,
                #                                filter_size=1,
                #                                stride=1,
                #                                use_dcn=self.use_dcn,
                #                                norm_type=self.norm_type)
                #                            )
                # conv_pre_feat.add_sublayer(
                #     conv_feat_name + '.{}_act2'.format(i), nn.GELU())
                self.add_sublayer('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)
            else:
                for j in range(end_level + 1 - i):
                    if j == 0:
                        ch_in = self.mid_channels
                    else:
                        ch_in = self.mid_channels
                    conv_pre_feat.add_sublayer(conv_feat_name + '.conv' + str(j),
                                               ConvNormLayer(
                                                   ch_in=ch_in,
                                                   ch_out=self.mid_channels,
                                                   filter_size=3 + 2 * j if self.use_same_kernels is False else 3,
                                                   stride=1,
                                                   use_dcn=self.use_dcn,
                                                   norm_type=self.norm_type))
                    conv_pre_feat.add_sublayer(
                        conv_feat_name + '.{}_act'.format(j), nn.GELU())
                conv_pre_feat.add_sublayer(conv_feat_name + '.{}_upsample'.format(i), nn.Upsample(scale_factor=2, mode='bilinear'))
                self.add_sublayer('conv_pre_feat' + str(i), conv_pre_feat)
                self.convs_all_levels.append(conv_pre_feat)

        self.conv_feat = []
        for i in range(self.range_level + 1):
            conv_feat_name = 'MaskHead.conv_feat{}'.format(i)
            if i == self.range_level:
                conv_feat = self.add_sublayer(conv_feat_name + '.conv',
                                              ConvNormLayer(
                                                  ch_in=self.in_channels + 2,
                                                  ch_out=self.mid_channels,
                                                  filter_size=3,
                                                  stride=1,
                                                  norm_type=self.norm_type))
            else:
                conv_feat = self.add_sublayer(conv_feat_name + '.conv',
                                              ConvNormLayer(
                                                  ch_in=self.in_channels,
                                                  ch_out=self.mid_channels,
                                                  filter_size=3,
                                                  stride=1,
                                                  norm_type=self.norm_type))
            self.conv_feat.append(conv_feat)

        conv_pred_name = 'MaskHead.conv_pred'
        self.convs_pre_feat = self.add_sublayer(conv_pred_name + '.0',
                                                ConvNormLayer(
                                                    ch_in=self.mid_channels,
                                                    ch_out=self.out_channels,
                                                    filter_size=1,
                                                    norm_type='bn',
                                                    stride=1,
                                                    use_dcn=False))
        self.convs_pre = self.add_sublayer(conv_pred_name + '.1',
                                           ConvNormLayer(
                                               ch_in=self.out_channels,
                                               ch_out=self.out_channels,
                                               filter_size=1,
                                               stride=1,
                                               use_dcn=False,
                                               norm_type=self.norm_type)
                                           )

        enhance_reverse_attention = []
        for i in range(2):
            the_reverse_vision = self.add_sublayer('enhance_reverse_attention.block{}'.format(i), ReverseVision(in_channels=self.mid_channels, out_channels=1))
            enhance_reverse_attention.append(the_reverse_vision)
        self.enhance_reverse_attention = enhance_reverse_attention

        self.conv_out = []
        for i in range(self.range_level + 1):
            conv_out_name = 'MaskHead.conv_out{}'.format(i)
            conv_out = nn.Sequential()
            for conv_i in range(i):
                if i == self.range_level and conv_i == 0:
                    ch_in = self.mid_channels
                elif i is not self.range_level and conv_i == 0:
                    ch_in = self.in_channels
                else:
                    ch_in = self.mid_channels
                conv_out.add_sublayer(conv_out_name + '.{}'.format(conv_i),
                                      ConvNormLayer(
                                          ch_in=ch_in,
                                          ch_out=self.mid_channels,
                                          filter_size=1 if i == self.range_level else 3,
                                          stride=1,
                                          use_dcn=False,
                                          norm_type=self.norm_type)
                                      )
                conv_out.add_sublayer(conv_out_name + '.act{}'.format(conv_i), nn.ReLU())
            if i == 0:
                conv_out.add_sublayer(conv_out_name + '.0.conv1',
                                      ConvNormLayer(
                                          ch_in=self.in_channels,
                                          ch_out=self.mid_channels,
                                          filter_size=1,
                                          stride=1,
                                          use_dcn=False,
                                          norm_type=self.norm_type)
                                      )
                conv_out.add_sublayer(conv_out_name + '.0.act1', nn.GELU())
                conv_out.add_sublayer(conv_out_name + '.0.conv2',
                                      ConvNormLayer(
                                          ch_in=self.mid_channels,
                                          ch_out=self.mid_channels,
                                          filter_size=3,
                                          stride=1,
                                          use_dcn=False,
                                          norm_type=self.norm_type)
                                      )
                conv_out.add_sublayer(conv_out_name + '.0.act2', nn.GELU())
            # conv_out.add_sublayer(conv_out_name + ".last_conv",
            #                       Conv2d(
            #                           in_channels=self.mid_channels,
            #                           out_channels=self.out_channels,
            #                           kernel_size=1,
            #                           stride=1,
            #                           bias=True,
            #                           weight_init=KaimingNormal())
            #                       )
            conv_out.add_sublayer(conv_out_name + ".last_conv",
                                  Conv2d(
                                      in_channels=self.mid_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      bias=False,
                                      weight_init=KaimingNormal())
                                  )
            self.add_sublayer('conv_out_level{}_seq'.format(i), conv_out)
            self.conv_out.append(conv_out)

        self.conv_out_feat = self.add_sublayer('conv_out_feat',
                                               ConvNormLayer(ch_in=self.in_channels,
                                                             ch_out=self.mid_channels,
                                                             filter_size=1,
                                                             stride=1,
                                                             norm_type='gn')
                                               )
        self.out_feat_dim = []
        for i in range(self.range_level + 1):
            out_feat_dim_name = 'out_feat_dim.{}'.format(i)
            out_feat_dim = self.add_sublayer(out_feat_dim_name + '.conv',
                                             ConvNormLayer(ch_in=self.out_channels,
                                                           ch_out=self.mid_channels,
                                                           filter_size=3 if i == self.range_level else 3,
                                                           stride=1,
                                                           norm_type='gn')
                                             )
            self.out_feat_dim.append(out_feat_dim)

        self.slice_feat = SuperResolutionBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, tem_feats, boundary_feat):
        out_feats = []
        edge_maps = []
        # 自上而下
        for i in range(self.range_level, -1, -1):
            if i == self.range_level:
                # TODO:加入一致性卷积
                input_feat = tem_feats[i]
                x_range = paddle.linspace(
                    -1, 1, paddle.shape(input_feat)[-1], dtype='float32')
                y_range = paddle.linspace(
                    -1, 1, paddle.shape(input_feat)[-2], dtype='float32')
                y, x = paddle.meshgrid([y_range, x_range])
                x = paddle.unsqueeze(x, [0, 1])
                y = paddle.unsqueeze(y, [0, 1])
                y = paddle.expand(
                    y, shape=[paddle.shape(input_feat)[0], 1, -1, -1])
                x = paddle.expand(
                    x, shape=[paddle.shape(input_feat)[0], 1, -1, -1])
                coord_feat = paddle.concat([x, y], axis=1)

                out_feat = self.conv_out_feat(tem_feats[i])  # 调整维度->128
                out_feat = F.gelu(out_feat)  # 激活tem_i特征图
                out_feat = self.conv_out[i](out_feat)
                out_feats.append(out_feat)  # feat_all_level:out_feats影响卷积核和分类器，影响分类loss。最后一步进行全连接（即1x1conv）

                out_feat = self.out_feat_dim[i](out_feat)  # TODO:CB,需要激活一下
                if self.boundary_learn:
                    boundary_feat = F.sigmoid(boundary_feat)
                    edge_maps.append(boundary_feat)  # save boundary map
                    boundary_feat = F.interpolate(boundary_feat, scale_factor=0.25, mode='bilinear')
                    weight_boundary = F.gelu(tem_feats[i]) * boundary_feat  # C:64, [0.5,1]弱化除边缘以外的各项特征
                else:
                    weight_boundary = F.gelu(tem_feats[i])
                weight_boundary = F.gelu(self.conv_feat[i](paddle.concat([weight_boundary, coord_feat], 1)))
                feat_all_level = paddle.add(weight_boundary, F.gelu(out_feat))  # 保证向下一阶段传递的目标明确
                # 替换：
                feat_all_level = self.convs_all_levels[i](feat_all_level)
                feat_all_level = F.gelu(feat_all_level)  # C:128 TODO:不需要
                feat_all_level = F.interpolate(feat_all_level, scale_factor=2., mode='bilinear')  # 1/32->1/16
            else:
                out_feat = self.conv_out[i](tem_feats[i])
                out_feats.append(out_feat)  # TODO:区别在于没有BN,给C3阶段的out_feats补充多尺度卷积

                # edge_map = self.conv_feat[i](tem_feats[i])  # 用变量承接一下
                out_feat = self.out_feat_dim[i](out_feat)
                feat_all_level = paddle.add(F.gelu(out_feat), feat_all_level)  # feat_all_level will be here
                feat_all_level = self.convs_all_levels[i](feat_all_level)
                # TODO:reverse_boundary_map
                if self.boundary_learn:
                    edge_map = self.enhance_reverse_attention[i](out_feat, feat_all_level)
                    edge_maps.append(F.sigmoid(edge_map))
        feat_all_level = self.convs_pre_feat(feat_all_level)  # out_channel + 0
        feat_all_level = F.gelu(feat_all_level * 3)
        # TODO:对C2/C4阶段的特征图进行切分和重组
        # slice_feat = self.slice_feat(feat_all_level)
        # feat_all_level = paddle.concat([feat_all_level, slice_feat], axis=1)
        # seg_pred = F.gelu(self.convs_pre(feat_all_level))  # C:256
        seg_pred = feat_all_level
        return seg_pred, list(reversed(out_feats)), edge_maps

    @classmethod
    def from_config(cls, cfg, input_shape):
        print('TransMaskHead:\n\tin_channels:{}'.format(input_shape[0].channels))
        return {
            'in_channels': input_shape[0].channels,
        }


@register
class ReverseVision(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels=128,
                 mid_channels=128,
                 out_channels=1,
                 start_level=0,
                 end_level=3,
                 use_dcn_in_tower=False,
                 norm_type='gn'):
        super(ReverseVision, self).__init__()

        attention_seq_name = "reverse_vision"
        attention_seq = nn.Sequential()
        attention_seq.add_sublayer(attention_seq_name + '.conv2d',
                                   Conv2d(in_channels=2,
                                          out_channels=1,
                                          kernel_size=7,
                                          padding=3,
                                          bias=False,
                                          weight_init=KaimingNormal())
                                   )
        attention_seq.add_sublayer(attention_seq_name + '.act',
                                   nn.Sigmoid())
        self.add_sublayer(attention_seq_name, attention_seq)
        self.conv1 = attention_seq
        self.edge_seg = self.add_sublayer("reverse_vision.edge_seg",
                                          Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 stride=1,
                                                 bias=False,
                                                 weight_init=KaimingNormal())
                                          )

    def forward(self, tem_feat, main_feat):
        tem_feat = F.interpolate(tem_feat, scale_factor=2., mode='bilinear')
        avg_x = paddle.mean(tem_feat, axis=1, keepdim=True)
        max_x = paddle.max(tem_feat, axis=1, keepdim=True)
        x = paddle.concat([avg_x, max_x], 1)
        x = 1 - self.conv1(x)  # self.conv1 include sigmoid
        x = main_feat * x  # main_feat语义信息强烈

        x = self.edge_seg(x)
        return x


@register
class SuperResolutionBlock(nn.Layer):
    __shared__ = ['norm_type']

    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 slice_num=2,
                 slice_scale_ratio=2):
        super(SuperResolutionBlock, self).__init__()
        self.slice_num = slice_num
        self.slice_scale_ratio = slice_scale_ratio
        self.spatial_attention_conv = self.add_sublayer('SuperResolutionBlock.spatial_attention_conv',
                                                        Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=3 // 2, bias=False, weight_init=KaimingNormal()))
        self.conv1 = self.add_sublayer('SuperResolutionBlock.attention_conv',
                                       ConvNormLayer(
                                           ch_in=in_channels,
                                           ch_out=out_channels,
                                           filter_size=1,
                                           stride=1,
                                           norm_type='bn')
                                       )
        conv_slice = nn.Sequential()
        conv_slice.add_sublayer('SuperResolutionBlock.conv_slice.conv1',
                                ConvNormLayer(
                                    ch_in=out_channels,
                                    ch_out=out_channels,
                                    filter_size=3,
                                    stride=1,
                                    # padding=1,
                                    norm_type='bn')
                                )
        conv_slice.add_sublayer('SuperResolutionBlock.conv_slice.act1', nn.ReLU())
        conv_slice.add_sublayer('SuperResolutionBlock.conv_slice.conv2',
                                ConvNormLayer(
                                    ch_in=out_channels,
                                    ch_out=out_channels,
                                    filter_size=3,
                                    stride=1,
                                    # padding=1,
                                    norm_type='bn')
                                )
        conv_slice.add_sublayer('SuperResolutionBlock.conv_slice.act2', nn.ReLU())
        self.add_sublayer('SuperResolutionBlock.conv_slice', conv_slice)
        self.conv_slice = conv_slice

        # self.channel_attention = ChannelAttention(in_channels)

    def spatial_attention(self, feat):
        x_max = paddle.max(feat, axis=1, keepdim=True)
        y_avg = paddle.mean(feat, axis=1, keepdim=True)
        # TODO：未来加一层一致性卷积
        out = paddle.concat([x_max, y_avg], axis=1)
        out = self.spatial_attention_conv(out)
        out_weights = F.sigmoid(out)
        return out_weights * feat

    def forward(self, input):
        reshape_size_N, reshape_size_C, reshape_size_h, reshape_size_w = input.shape
        # [2,256,H,W]
        output = []
        y = paddle.chunk(input, self.slice_num, axis=2)  # [2,256,H/4,W]
        slice_result = []
        for item in y:
            # res = paddle.chunk(item, 4, axis=-1)  # [2,256,H/4,W/4]
            slice_result.append(paddle.chunk(item, self.slice_num, axis=-1))
        # 卷积
        for i, rows in enumerate(slice_result):
            for j, col_tensor in enumerate(rows):
                # upsample
                field_windows_h = reshape_size_h / self.slice_num * self.slice_scale_ratio
                field_windows_w = reshape_size_w / self.slice_num * self.slice_scale_ratio
                field_windows = F.interpolate(col_tensor, size=[field_windows_h, field_windows_w], mode='bilinear', align_corners=True)  # 放大
                # field_attention_map = self.spatial_attention(field_windows)
                field_windows = F.gelu(self.conv1(field_windows))  # 调整分布
                # field_windows = paddle.add(self.spatial_attention(field_windows), field_windows)
                field_windows = self.spatial_attention(field_windows)
                field_windows = F.max_pool2d(field_windows, kernel_size=7, stride=2, padding=3)
                field_windows = F.interpolate(field_windows, size=[reshape_size_h / self.slice_num, reshape_size_w / self.slice_num], mode='bilinear', align_corners=True)
                slice_result[i][j] = field_windows
        # 拼回来
        output = []
        for item in slice_result:
            # out_cols = paddle.concat(item, axis=-1)
            output.append(paddle.concat(item, axis=-1))
        slice_result = paddle.concat(output, axis=-2)
        slice_result = self.conv_slice(slice_result)
        return slice_result


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
        out = paddle.concat([x_max, y_avg], axis=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return paddle.multiply(inputs, out)


@register
@serializable
class FPN_BS(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channel,
                 spatial_scales=[0.25, 0.125, 0.0625, 0.03125],
                 has_extra_convs=False,
                 extra_stage=1,
                 use_c5=True,
                 norm_type=None,
                 norm_decay=0.,
                 freeze_norm=False,
                 relu_before_extra_convs=True):
        super(FPN_BS, self).__init__()
        print("--------------------using fpn with boundary supervised module --------------------")
        print("--------------------in_channels:{}, out_channel:{} --------------------".format(in_channels,out_channel))
        self.out_channel = out_channel
        for s in range(extra_stage):
            spatial_scales = spatial_scales + [spatial_scales[-1] / 2.]
        self.spatial_scales = spatial_scales
        self.has_extra_convs = has_extra_convs
        self.extra_stage = extra_stage
        self.use_c5 = use_c5
        self.relu_before_extra_convs = relu_before_extra_convs
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.freeze_norm = freeze_norm
        self.fpn_module = FPN(in_channels,
                              out_channel,
                              spatial_scales=spatial_scales,
                              has_extra_convs=has_extra_convs,
                              extra_stage=extra_stage,
                              use_c5=use_c5,
                              norm_type=norm_type,
                              norm_decay=norm_decay,
                              freeze_norm=freeze_norm,
                              relu_before_extra_convs=relu_before_extra_convs)
        self.boundary_conv = Boundary_Attention(in_channel=out_channel, out_channel=1)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'spatial_scales': [1.0 / i.stride for i in input_shape],
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=1. / s)
            for s in self.spatial_scales
        ]

    def forward(self, inputs):
        x = self.fpn_module(inputs)  # 提取C3\C4\C5,不提取C6
        boundary_map = self.boundary_conv(x, inputs[0])  # 用x[0],x[1],x[2]和inputs[0]生成边缘
        return x, boundary_map
