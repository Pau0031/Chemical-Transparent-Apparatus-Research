# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant

from ppdet.modeling.layers import ConvNormLayer, MaskMatrixNMS, DropBlock, Conv2d, SeparableConvBNReLU, ConvBNReLU
from ppdet.core.workspace import register

from six.moves import zip
import numpy as np

__all__ = ['TransSOLOv2Head']


@register
class TransSOLOv2Head(nn.Layer):
    """
    Head block for SOLOv2 network

    Args:
        num_classes (int): Number of output classes.
        in_channels (int): Number of input channels.
        seg_feat_channels (int): Num_filters of kernel & categroy branch convolution operation.
        stacked_convs (int): Times of convolution operation.
        num_grids (list[int]): List of feature map grids size.
        kernel_out_channels (int): Number of output channels in kernel branch.
        dcn_v2_stages (list): Which stage use dcn v2 in tower. It is between [0, stacked_convs).
        segm_strides (list[int]): List of segmentation area stride.
        solov2_loss (object): SOLOv2Loss instance.
        score_threshold (float): Threshold of categroy score.
        mask_nms (object): MaskMatrixNMS instance.
    """
    __inject__ = ['solov2_loss', 'mask_nms']
    __shared__ = ['norm_type', 'num_classes']

    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 num_grids=[40, 36, 24, 16, 12],
                 kernel_out_channels=256,
                 dcn_v2_stages=[],
                 segm_strides=[8, 8, 16, 32, 32],
                 solov2_loss=None,
                 score_threshold=0.1,
                 mask_threshold=0.5,
                 mask_nms=None,
                 norm_type='gn',
                 drop_block=False):
        super(TransSOLOv2Head, self).__init__()
        print("--------------------using num_classes:{} --------------------".format(num_grids))
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = kernel_out_channels
        self.dcn_v2_stages = dcn_v2_stages
        self.segm_strides = segm_strides
        self.solov2_loss = solov2_loss
        self.mask_nms = mask_nms
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.norm_type = norm_type
        self.drop_block = drop_block

        self.kernel_pred_convs = []
        self.cate_pred_convs = []
        for i in range(self.stacked_convs):
            use_dcn = True if i in self.dcn_v2_stages else False
            ch_in = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            kernel_conv = self.add_sublayer(
                'bbox_head.kernel_convs.' + str(i),
                ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    use_dcn=use_dcn,
                    norm_type=self.norm_type))
            self.kernel_pred_convs.append(kernel_conv)
            ch_in = self.in_channels if i == 0 else self.seg_feat_channels
            cate_conv = self.add_sublayer(
                'bbox_head.cate_convs.' + str(i),
                ConvNormLayer(
                    ch_in=ch_in,
                    ch_out=self.seg_feat_channels,
                    filter_size=3,
                    stride=1,
                    use_dcn=use_dcn,
                    norm_type=self.norm_type))
            self.cate_pred_convs.append(cate_conv)

        self.solo_kernel = self.add_sublayer(
            'bbox_head.solo_kernel',
            nn.Conv2D(
                self.seg_feat_channels,
                self.kernel_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=True))
        self.solo_cate = self.add_sublayer(
            'bbox_head.solo_cate',
            nn.Conv2D(
                self.seg_feat_channels,
                self.cate_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(
                    value=float(-np.log((1 - 0.01) / 0.01))))))

        if self.drop_block and self.training:
            self.drop_block_fun = DropBlock(
                block_size=3, keep_prob=0.9, name='solo_cate.dropblock')

    def _points_nms(self, heat, kernel_size=2):
        hmax = F.max_pool2d(heat, kernel_size=kernel_size, stride=1, padding=1)
        keep = paddle.cast((hmax[:, :, :-1, :-1] == heat), 'float32')
        return heat * keep

    def _split_feats(self, feats):
        return (F.interpolate(
            feats[0],
            scale_factor=0.5,
            align_corners=False,
            align_mode=0,
            mode='bilinear'), feats[1], feats[2], feats[3], F.interpolate(
            feats[4],
            size=paddle.shape(feats[3])[-2:],
            mode='bilinear',
            align_corners=False,
            align_mode=0))

    def forward(self, input):
        """
        Get SOLOv2 head output

        Args:
            input (list): List of Tensors, output of backbone or neck stages
        Returns:
            cate_pred_list (list): Tensors of each category branch layer
            kernel_pred_list (list): Tensors of each kernel branch layer
        """
        # 对input各阶段的分辨率做了调整
        # feats = self._split_feats(input)
        feats = input  # 不需要对分辨率进行调整
        cate_pred_list = []
        kernel_pred_list = []
        # seg_num_grids的长度 与 所取的resnet阶段总数一致
        for idx in range(len(self.seg_num_grids)):
            cate_pred, kernel_pred = self._get_output_single(feats[idx], idx)
            cate_pred_list.append(cate_pred)
            kernel_pred_list.append(kernel_pred)

        return cate_pred_list, kernel_pred_list

    def _get_output_single(self, input, idx):
        ins_kernel_feat = input
        # CoordConv
        x_range = paddle.linspace(
            -1, 1, paddle.shape(ins_kernel_feat)[-1], dtype='float32')
        y_range = paddle.linspace(
            -1, 1, paddle.shape(ins_kernel_feat)[-2], dtype='float32')
        y, x = paddle.meshgrid([y_range, x_range])
        x = paddle.unsqueeze(x, [0, 1])
        y = paddle.unsqueeze(y, [0, 1])
        y = paddle.expand(
            y, shape=[paddle.shape(ins_kernel_feat)[0], 1, -1, -1])
        x = paddle.expand(
            x, shape=[paddle.shape(ins_kernel_feat)[0], 1, -1, -1])
        coord_feat = paddle.concat([x, y], axis=1)
        ins_kernel_feat = paddle.concat([ins_kernel_feat, coord_feat], axis=1)

        # kernel branch
        kernel_feat = ins_kernel_feat  # [N,258,H,W]
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(
            kernel_feat,
            size=[seg_num_grid, seg_num_grid],
            mode='bilinear',
            align_corners=False,
            align_mode=0)
        cate_feat = kernel_feat[:, :-2, :, :]  # 该维度最后2个图是用于坐标的。

        for kernel_layer in self.kernel_pred_convs:
            kernel_feat = F.relu(kernel_layer(kernel_feat))  # [N,512,H,W] 备注：连续多维度卷积所以玄学，看似越无关的kernel通过非线性计算，达到看似无关，其实相关的目的。
        if self.drop_block and self.training:
            kernel_feat = self.drop_block_fun(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)
        # cate branch
        for cate_layer in self.cate_pred_convs:
            cate_feat = F.relu(cate_layer(cate_feat))
        if self.drop_block and self.training:
            cate_feat = self.drop_block_fun(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if not self.training:
            cate_pred = self._points_nms(F.sigmoid(cate_pred), kernel_size=2)
            cate_pred = paddle.transpose(cate_pred, [0, 2, 3, 1])
        return cate_pred, kernel_pred

    def get_loss(self, cate_preds, kernel_preds, ins_pred, ins_labels,
                 cate_labels, grid_order_list, fg_num, gt_boundary_labels, boundary_pred):
        """
        Get loss of network of SOLOv2.

        Args:
            cate_preds (list): Tensor list of categroy branch output.
            kernel_preds (list): Tensor list of kernel branch output.
            ins_pred (list): Tensor list of instance branch output.
            ins_labels (list): List of instance labels pre batch.
            cate_labels (list): List of categroy labels pre batch.
            grid_order_list (list): List of index in pre grid.
            fg_num (int): Number of positive samples in a mini-batch.
            gt_boundary_labels (list): List of boundary labels pre batch.
        Returns:
            loss_ins (Tensor): The instance loss Tensor of SOLOv2 network.
            loss_cate (Tensor): The category loss Tensor of SOLOv2 network.
            loss_boundary (Tensor): The boundary loss Tensor of SOLOv2 network.
        """
        batch_size = paddle.shape(grid_order_list[0])[0]
        ins_pred_list = []
        for kernel_preds_level, grid_orders_level in zip(kernel_preds,
                                                         grid_order_list):
            if grid_orders_level.shape[1] == 0:
                ins_pred_list.append(None)
                continue
            grid_orders_level = paddle.reshape(grid_orders_level, [-1])  # 同一个batch的grid_order合并
            reshape_pred = paddle.reshape(
                kernel_preds_level,
                shape=(paddle.shape(kernel_preds_level)[0],
                       paddle.shape(kernel_preds_level)[1], -1))
            reshape_pred = paddle.transpose(reshape_pred, [0, 2, 1])
            reshape_pred = paddle.reshape(
                reshape_pred, shape=(-1, paddle.shape(reshape_pred)[2]))  # 同一个batch的kernel_preds_level合并：[N*H*W,C]
            gathered_pred = paddle.gather(reshape_pred, index=grid_orders_level)  # 选取行（对应的序号的卷积核），axis默认0
            gathered_pred = paddle.reshape(
                gathered_pred,
                shape=[batch_size, -1, paddle.shape(gathered_pred)[1]])  # [N,卷积核的数量,C]
            cur_ins_pred = ins_pred  # [N,C,H,W]
            cur_ins_pred = paddle.reshape(
                cur_ins_pred,
                shape=(paddle.shape(cur_ins_pred)[0],
                       paddle.shape(cur_ins_pred)[1], -1))  # 合并H,W:[N,C,H*W]
            ins_pred_conv = paddle.matmul(gathered_pred, cur_ins_pred)
            cur_ins_pred = paddle.reshape(
                ins_pred_conv,
                shape=(-1, paddle.shape(ins_pred)[-2],
                       paddle.shape(ins_pred)[-1]))  # 复原：[N*C,H,W]
            ins_pred_list.append(cur_ins_pred)

        num_ins = paddle.sum(fg_num)
        cate_preds = [
            paddle.reshape(
                paddle.transpose(cate_pred, [0, 2, 3, 1]),
                shape=(-1, self.cate_out_channels)) for cate_pred in cate_preds
        ]  # 合并C,H,W
        flatten_cate_preds = paddle.concat(cate_preds)  # 准备好一个batch预测的类别
        new_cate_labels = []
        for cate_label in cate_labels:
            new_cate_labels.append(paddle.reshape(cate_label, shape=[-1]))
        cate_labels = paddle.concat(new_cate_labels)  # 合并，并准备好一个batch的GT类别

        for i in range(len(boundary_pred)):
            boundary_pred[i] = F.interpolate(boundary_pred[i], paddle.shape(gt_boundary_labels[0])[-2:], mode='bilinear')  # 调整分辨率
        loss_ins, loss_cate, loss_boundaries, focal_loss_ins = self.solov2_loss(
            ins_pred_list, ins_labels, flatten_cate_preds, cate_labels, num_ins, gt_boundary_labels, boundary_pred)

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate, "loss_boundaries": loss_boundaries, "focal_loss_ins": focal_loss_ins}

    def get_prediction(self, cate_preds, kernel_preds, seg_pred, im_shape,
                       scale_factor):
        """
        Get prediction result of SOLOv2 network

        Args:
            cate_preds (list): List of Variables, output of categroy branch.
            kernel_preds (list): List of Variables, output of kernel branch.
            seg_pred (list): List of Variables, output of mask head stages.
            im_shape (Variables): [h, w] for input images.
            scale_factor (Variables): [scale, scale] for input images.
        Returns:
            seg_masks (Tensor): The prediction segmentation.
            cate_labels (Tensor): The prediction categroy label of each segmentation.
            seg_masks (Tensor): The prediction score of each segmentation.
        """
        num_levels = len(cate_preds)
        featmap_size = paddle.shape(seg_pred)[-2:]
        seg_masks_list = []
        cate_labels_list = []
        cate_scores_list = []
        cate_preds = [cate_pred * 1.0 for cate_pred in cate_preds]
        kernel_preds = [kernel_pred * 1.0 for kernel_pred in kernel_preds]
        # Currently only supports batch size == 1
        for idx in range(1):
            cate_pred_list = [
                paddle.reshape(
                    cate_preds[i][idx], shape=(-1, self.cate_out_channels))
                for i in range(num_levels)
            ]
            seg_pred_list = seg_pred
            kernel_pred_list = [
                paddle.reshape(
                    paddle.transpose(kernel_preds[i][idx], [1, 2, 0]),
                    shape=(-1, self.kernel_out_channels))
                for i in range(num_levels)
            ]
            cate_pred_list = paddle.concat(cate_pred_list, axis=0)
            kernel_pred_list = paddle.concat(kernel_pred_list, axis=0)

            seg_masks, cate_labels, cate_scores = self.get_seg_single(
                cate_pred_list, seg_pred_list, kernel_pred_list, featmap_size,
                im_shape[idx], scale_factor[idx][0])
            bbox_num = paddle.shape(cate_labels)[0]
        return seg_masks, cate_labels, cate_scores, bbox_num

    def get_seg_single(self, cate_preds, seg_preds, kernel_preds, featmap_size,
                       im_shape, scale_factor):
        """
        The code of this function is based on:
            https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solov2_head.py#L385
        """
        h = paddle.cast(im_shape[0], 'int32')[0]
        w = paddle.cast(im_shape[1], 'int32')[0]
        upsampled_size_out = [featmap_size[0] * 4, featmap_size[1] * 4]

        y = paddle.zeros(shape=paddle.shape(cate_preds), dtype='float32')
        inds = paddle.where(cate_preds > self.score_threshold, cate_preds, y)
        inds = paddle.nonzero(inds)
        cate_preds = paddle.reshape(cate_preds, shape=[-1])
        # Prevent empty and increase fake data
        ind_a = paddle.cast(paddle.shape(kernel_preds)[0], 'int64')
        ind_b = paddle.zeros(shape=[1], dtype='int64')
        inds_end = paddle.unsqueeze(paddle.concat([ind_a, ind_b]), 0)
        inds = paddle.concat([inds, inds_end])
        kernel_preds_end = paddle.ones(
            shape=[1, self.kernel_out_channels], dtype='float32')
        kernel_preds = paddle.concat([kernel_preds, kernel_preds_end])
        cate_preds = paddle.concat(
            [cate_preds, paddle.zeros(
                shape=[1], dtype='float32')])

        # cate_labels & kernel_preds
        cate_labels = inds[:, 1]
        kernel_preds = paddle.gather(kernel_preds, index=inds[:, 0])
        cate_score_idx = paddle.add(inds[:, 0] * self.cate_out_channels,
                                    cate_labels)  # i*cols_num+j
        cate_scores = paddle.gather(cate_preds, index=cate_score_idx)

        size_trans = np.power(self.seg_num_grids, 2)
        strides = []
        for _ind in range(len(self.segm_strides)):
            strides.append(
                paddle.full(
                    shape=[int(size_trans[_ind])],
                    fill_value=self.segm_strides[_ind],
                    dtype="int32"))
        strides = paddle.concat(strides)
        strides = paddle.concat(
            [strides, paddle.zeros(
                shape=[1], dtype='int32')])
        strides = paddle.gather(strides, index=inds[:, 0])

        # mask encoding.
        kernel_preds = paddle.unsqueeze(kernel_preds, [2, 3])
        seg_preds = F.conv2d(seg_preds, kernel_preds)  # 1*1卷积生成pred_mask
        seg_preds = F.sigmoid(paddle.squeeze(seg_preds, [0]))  # 生成0-1概率矩阵
        seg_masks = seg_preds > self.mask_threshold  # 生成0/1二值mask
        seg_masks = paddle.cast(seg_masks, 'float32')
        sum_masks = paddle.sum(seg_masks, axis=[1, 2])

        y = paddle.zeros(shape=paddle.shape(sum_masks), dtype='float32')
        keep = paddle.where(sum_masks > strides, sum_masks, y)
        keep = paddle.nonzero(keep)
        keep = paddle.squeeze(keep, axis=[1])
        # Prevent empty and increase fake data
        keep_other = paddle.concat(
            [keep, paddle.cast(paddle.shape(sum_masks)[0] - 1, 'int64')])
        keep_scores = paddle.concat(
            [keep, paddle.cast(paddle.shape(sum_masks)[0], 'int64')])
        cate_scores_end = paddle.zeros(shape=[1], dtype='float32')
        cate_scores = paddle.concat([cate_scores, cate_scores_end])

        seg_masks = paddle.gather(seg_masks, index=keep_other)
        seg_preds = paddle.gather(seg_preds, index=keep_other)
        sum_masks = paddle.gather(sum_masks, index=keep_other)
        cate_labels = paddle.gather(cate_labels, index=keep_other)
        cate_scores = paddle.gather(cate_scores, index=keep_scores)

        # mask scoring.
        seg_mul = paddle.cast(seg_preds * seg_masks, 'float32')
        seg_scores = paddle.sum(seg_mul, axis=[1, 2]) / sum_masks
        cate_scores *= seg_scores
        # Matrix NMS
        seg_preds, cate_scores, cate_labels = self.mask_nms(
            seg_preds, seg_masks, cate_labels, cate_scores, sum_masks=sum_masks)  # 输出最终的类别结果和置信度
        ori_shape = im_shape[:2] / scale_factor + 0.5
        ori_shape = paddle.cast(ori_shape, 'int32')
        seg_preds = F.interpolate(
            paddle.unsqueeze(seg_preds, 0),
            size=upsampled_size_out,
            mode='bilinear',
            align_corners=False,
            align_mode=0)
        seg_preds = paddle.slice(
            seg_preds, axes=[2, 3], starts=[0, 0], ends=[h, w])
        seg_masks = paddle.squeeze(
            F.interpolate(
                seg_preds,
                size=ori_shape[:2],
                mode='bilinear',
                align_corners=False,
                align_mode=0),
            axis=[0])
        seg_masks = paddle.cast(seg_masks > self.mask_threshold, 'uint8')  # 由概率矩阵生成最终的二值seg_mask
        return seg_masks, cate_labels, cate_scores
