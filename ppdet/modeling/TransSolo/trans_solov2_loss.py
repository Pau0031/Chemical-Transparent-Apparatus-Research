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
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
import cv2
import numpy as np

__all__ = ['TransSOLOv2Loss']


@register
@serializable
class TransSOLOv2Loss(object):
    """
    SOLOv2Loss
    Args:
        ins_loss_weight (float): Weight of instance loss.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        focal_loss_alpha (float): Alpha parameter for focal loss.
    """

    def __init__(self,
                 ins_loss_weight=3.0,
                 focal_loss_gamma=2.0,
                 focal_loss_alpha=0.25):
        self.ins_loss_weight = ins_loss_weight
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        # TODO: boundary_loss_wight
        self.boundary_loss_wight = 1.0

    def _dice_loss(self, input, target):
        input = paddle.reshape(input, shape=(paddle.shape(input)[0], -1))  # 展平H,W
        target = paddle.reshape(target, shape=(paddle.shape(target)[0], -1))
        a = paddle.sum(input * target, axis=1)
        b = paddle.sum(input * input, axis=1) + 0.001
        c = paddle.sum(target * target, axis=1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def _focal_loss(self, inputs, targets, weights):
        # shape:C,H,W
        # 输入数据 logit 一般是卷积层的输出，不需要经过 sigmoid 层。数据类型是 float32、float64。
        positive_samples = paddle.to_tensor(targets.shape[0], dtype='float32')
        res = F.sigmoid_focal_loss(inputs, label=targets, reduction='mean')   # , normalizer=positive_samples
        return res

    def get_boundary(self, mask, thicky=8):
        tmp = mask.numpy().astype('uint8')  # [512,512]
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        return boundary

    def generate_boundary_weights(self, weight_x):
        # TODO:利用前景和背景（0-1）的异或运算边缘三合一
        fg_weight = paddle.where(weight_x >= 0.55, paddle.to_tensor(1.), paddle.to_tensor(0.))  # 前景权重
        bg_weight = paddle.where(weight_x >= 0.50, paddle.to_tensor(1.), paddle.to_tensor(0.))
        boundary_weight = paddle.logical_xor(bg_weight, fg_weight)
        boundary_weight = paddle.cast(boundary_weight, dtype='float32')
        return boundary_weight

    def __call__(self, ins_pred_list, ins_label_list, cate_preds, cate_labels,
                 num_ins, gt_boundary_labels, boundaries_pred_list):
        """
            Get loss of network of SOLOv2.
            Args:
                ins_pred_list (list): Variable list of instance branch output.
                ins_label_list (list): List of instance labels pre batch.
                cate_preds (list): Concat Variable list of categroy branch output.
                cate_labels (list): Concat list of categroy labels pre batch.
                num_ins (int): Number of positive samples in a mini-batch.
                gt_boundary_labels (list): List of boundary labels pre batch.
            Returns:
                loss_ins (Variable): The instance loss Variable of SOLOv2 network.
                loss_cate (Variable): The category loss Variable of SOLOv2 network.
                loss_boundary (Variable): The boundary loss Variable of SOLOv2 network.
            """

        # 1. Ues dice_loss to calculate instance loss
        loss_ins = []
        # 2. Ues focal_loss to calculate small instance loss
        focal_loss_ins = []
        # 3. Ues dice_loss to calculate boundary loss
        # loss_boundaries = []
        loss_boundaries = paddle.zeros(shape=[1], dtype='float32')
        total_weights = paddle.zeros(shape=[1], dtype='float32')
        # TODO: loss_boundary
        if len(boundaries_pred_list) > 0:
            gt_boundary_labels = paddle.concat(gt_boundary_labels, axis=1)  # [N,num,H,W]
            gt_boundary_labels = paddle.to_tensor(gt_boundary_labels, dtype="float32")  # [N,num,H,W]:[2,num,H,w]
            gt_boundary_labels = paddle.sum(gt_boundary_labels, axis=1, keepdim=True)  # 将一张图像上的每个实例边缘合并[N,1,H,W]:[2,1,H,w] [cv2.imwrite("boundary_labels.jpg",gt_boundary_labels[0].numpy()[0]*255)]
            gt_boundary_labels = paddle.where(gt_boundary_labels > 0, paddle.to_tensor([1.]), gt_boundary_labels)
            gt_boundary_labels = paddle.reshape(
                gt_boundary_labels,
                shape=[-1, paddle.shape(boundaries_pred_list[0])[-2], paddle.shape(boundaries_pred_list[0])[-1]])  # 向input对齐,合并B,C->[N,H,W]
            boundary_total_weights = paddle.zeros(shape=[1], dtype='float32')
            boundary_loss = []
            # TODO:3种边缘map进行逻辑运算，合成一张边缘map
            # boundaries_pred_list = [paddle.bitwise_xor(paddle.multiply(boundaries_pred_list[0], boundaries_pred_list[1]), boundaries_pred_list[2])]
            for input in boundaries_pred_list:
                if input is None:
                    continue
                weights = paddle.cast(
                    paddle.sum(gt_boundary_labels, axis=[1, 2]) > 0, 'float32')  # 符合条件的布尔值，转化为浮点数：1.

                # TODO：利用异或运算，计算边缘
                # input_boundary = self.generate_boundary_weights(input)
                input_boundary = paddle.reshape(input, shape=(-1, paddle.shape(input)[-2], paddle.shape(input)[-1]))  # 复原：[N*C,H,W]
                boundary_dice_out = paddle.multiply(self._dice_loss(input_boundary, gt_boundary_labels), weights)  # boundaries_pred_list里的tensor([N,B,H,W]*3)应该保持H,W一致
                boundary_total_weights += paddle.sum(weights)
                boundary_loss.append(boundary_dice_out)
            boundary_loss = paddle.sum(paddle.concat(boundary_loss)) / boundary_total_weights  # 平均边缘损失
            loss_boundaries = boundary_loss * self.boundary_loss_wight

        for input, target in zip(ins_pred_list, ins_label_list):
            if input is None:
                continue
            target = paddle.cast(target, 'float32')
            target = paddle.reshape(
                target,
                shape=[-1, paddle.shape(input)[-2], paddle.shape(input)[-1]])  # 向input对齐,合并B,C->[N,H,W]
            # select valid index from target
            valid_target_idx = paddle.sum(target, axis=[1, 2]) > 0
            weights = paddle.cast(valid_target_idx, 'float32')  # 符合条件的布尔值，转化为浮点数：1.
            # TODO：using Focal Loss to balance small object and others
            focal_out = self._focal_loss(input[valid_target_idx], target[valid_target_idx], weights)  # , focal_out_weight
            input = F.sigmoid(input)  # 对预测mask进行激活，已经是一个batch的数据了
            dice_out = paddle.multiply(self._dice_loss(input, target), weights)
            total_weights += paddle.sum(weights)
            loss_ins.append(dice_out)
            focal_loss_ins.append(focal_out)
            # focal_loss_weight += focal_out_weight
        focal_loss_ins = paddle.sum(paddle.concat(focal_loss_ins))   # focal loss 平均损失/ focal_loss_weight/ len(ins_label_list)
        focal_loss_ins = focal_loss_ins * 1.0  # 0.625
        loss_ins = paddle.sum(paddle.concat(loss_ins)) / total_weights  # dice loss 平均损失
        loss_ins = loss_ins * self.ins_loss_weight

        # 2. Ues sigmoid_focal_loss to calculate category loss
        # expand onehot labels
        num_classes = cate_preds.shape[-1]
        cate_labels_bin = F.one_hot(cate_labels, num_classes=num_classes + 1)
        cate_labels_bin = cate_labels_bin[:, 1:]  # 准备好GT类别的矩阵

        loss_cate = F.sigmoid_focal_loss(
            cate_preds,
            label=cate_labels_bin,
            normalizer=num_ins + 1.,
            gamma=self.focal_loss_gamma,
            alpha=self.focal_loss_alpha)
        return loss_ins, loss_cate, loss_boundaries, focal_loss_ins
