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

from ppdet.core.workspace import register, create
# from .meta_arch import BaseArch

__all__ = ['SOLOv2_ASPP', 'SOLOv2_NCD_GRA']

from ppdet.modeling import BaseArch


@register
class SOLOv2_ASPP(BaseArch):
    """
    SOLOv2 network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        solov2_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        neck (object): neck of network, such as feature pyramid network instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, solov2_head, mask_head, neck=None, ncd=None, aspp=None):
        super(SOLOv2_ASPP, self).__init__()
        print("---------------using SOLOv2_AASPP architectures---------------")
        self.backbone = backbone
        self.neck = neck
        self.solov2_head = solov2_head
        self.mask_head = mask_head

        # 添加的模块
        self.ncd = ncd
        self.aspp = aspp

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        solov2_head = create(cfg['solov2_head'], **kwargs)
        mask_head = create(cfg['mask_head'], **kwargs)  # MYSOLOv2MaskHead参数写死
        # 新增嵌入模块,参数已写死
        ncd = create(cfg['ncd'])
        aspp = create(cfg['aspp'])
        return {
            'backbone': backbone,
            'neck': neck,
            'solov2_head': solov2_head,
            'mask_head': mask_head,
            'ncd': ncd,
            'aspp': aspp
        }

    def model_arch(self):
        body_feats = self.backbone(self.inputs)

        # TODO:处理1/16的特征图，做空洞卷积
        third_feat = self.aspp(body_feats[2])  # 1024->256

        body_feats = self.neck(body_feats)  # output_shape==256*H*W
        # TODO:visual 1
        self.third_feat = third_feat
        # self.visual_list["neck_maps"] = [b.numpy() for b in body_feats]  # 4 stage maps
        # self.visual_list["third_feat"] = [b.numpy() for b in third_feat]  # c:256
        # self.visual_list["attention_maps"] = [b.numpy() for b in self.aspp.attention_maps]  # c:1024

        # TODO:对FPN的特征进行增强
        body_feats = self.ncd(body_feats, third_feat)  # 5 stage maps
        # body_feats = self.ncd(body_feats)  # without aspp module
        # TODO:visual 2
        # self.visual_list["new_neck_maps"] = [b.numpy() for b in body_feats]  # 5 stage maps

        # NCD 得出一张粗略的特征图c6
        # ncd_feat = self.ncd(body_feats)  # c:1

        # TODO:visual 3
        # self.visual_list["ncd_maps"] = [b.numpy() for b in ncd_feat]  # 1 chanel map

        # 然后用c6和body_feats的前四层进行SINet_head的计算。先处理兼容性问题，得出list=4的mask_body_feats，送到mask_head进行计算。
        # TODO：统一两个输入的通道数为128，这个pred_mask阶段参考语义分割的前沿成果 visual 4
        self.seg_pred = self.mask_head(body_feats)

        # self.seg_pred = self.mask_head(mask_feats)  # body_feats

        self.cate_pred_list, self.kernel_pred_list = self.solov2_head(
            body_feats)

    # 在这里做文章
    def get_loss(self, ):
        loss = {}
        # get gt_ins_labels, gt_cate_labels, etc.
        gt_ins_labels, gt_cate_labels, gt_grid_orders, gt_boundary_labels = [], [], [], []
        fg_num = self.inputs['fg_num']
        for i in range(len(self.solov2_head.seg_num_grids)):
            ins_label = 'ins_label{}'.format(i)
            if ins_label in self.inputs:
                gt_ins_labels.append(self.inputs[ins_label])
            cate_label = 'cate_label{}'.format(i)
            if cate_label in self.inputs:
                gt_cate_labels.append(self.inputs[cate_label])
            grid_order = 'grid_order{}'.format(i)
            if grid_order in self.inputs:
                gt_grid_orders.append(self.inputs[grid_order])
            # boundary_label = 'boundary_label{}'.format(i)
            # if boundary_label in self.inputs:
            #     gt_boundary_labels.append(self.inputs[boundary_label])

        loss_solov2 = self.solov2_head.get_loss(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            gt_ins_labels, gt_cate_labels, gt_grid_orders, fg_num)  # , gt_boundary_labels
        loss.update(loss_solov2)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        seg_masks, cate_labels, cate_scores, bbox_num = self.solov2_head.get_prediction(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "segm": seg_masks,
            "bbox_num": bbox_num,
            'cate_label': cate_labels,
            'cate_score': cate_scores
        }
        return outs


@register
class SOLOv2_NCD_GRA(BaseArch):
    """
    SOLOv2 network, see https://arxiv.org/abs/2003.10152

    Args:
        backbone (object): an backbone instance
        solov2_head (object): an `SOLOv2Head` instance
        mask_head (object): an `SOLOv2MaskHead` instance
        neck (object): neck of network, such as feature pyramid network instance
    """

    __category__ = 'architecture'

    def __init__(self, backbone, solov2_head, mask_head, neck=None, ncd=None, aspp=None):
        super(SOLOv2_NCD_GRA, self).__init__()
        print("---------------using SOLOv2_NCD_GRA architectures with MYSOLOv2MaskHead---------------")
        self.backbone = backbone
        self.neck = neck
        self.solov2_head = solov2_head
        self.mask_head = mask_head

        # 添加的模块
        self.ncd = ncd
        self.aspp = aspp

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'input_shape': neck.out_shape}
        solov2_head = create(cfg['solov2_head'], **kwargs)
        mask_head = create(cfg['mask_head'], **kwargs)  # MYSOLOv2MaskHead参数写死
        # 新增嵌入模块,参数已写死
        ncd = create(cfg['ncd'])
        # aspp = create(cfg['aspp'])
        return {
            'backbone': backbone,
            'neck': neck,
            'solov2_head': solov2_head,
            'mask_head': mask_head,
            'ncd': ncd,
            # 'aspp': aspp
        }

    def model_arch(self):
        body_feats = self.backbone(self.inputs)

        # TODO:处理1/16的特征图，做空洞卷积
        # third_feat = self.aspp(body_feats[2])  # 1024->256

        body_feats = self.neck(body_feats)  # output_shape==256*H*W
        # TODO:visual 1
        self.body_feats = body_feats
        # self.visual_list["neck_maps"] = [b.numpy() for b in body_feats]  # 4 stage maps
        # self.visual_list["third_feat"] = [b.numpy() for b in third_feat]  # c:256
        # self.visual_list["attention_maps"] = [b.numpy() for b in self.aspp.attention_maps]  # c:1024

        # TODO:对FPN的特征进行增强
        ncd_map = self.ncd(body_feats[3], body_feats[2], body_feats[1])  # 3 stage maps
        # TODO:visual 2
        self.ncd_map = ncd_map
        # self.visual_list["new_neck_maps"] = [b.numpy() for b in body_feats]  # 5 stage maps

        # NCD 得出一张粗略的特征图c6
        # ncd_feat = self.ncd(body_feats)  # c:1

        # TODO:visual 3
        # self.visual_list["ncd_maps"] = [b.numpy() for b in ncd_feat]  # 1 chanel map

        # 然后用c6和body_feats的前四层进行SINet_head的计算。先处理兼容性问题，得出list=4的mask_body_feats，送到mask_head进行计算。
        # TODO：统一两个输入的通道数为128，这个pred_mask阶段参考语义分割的前沿成果 visual 4
        self.seg_pred = self.mask_head(body_feats, ncd_map)

        # self.seg_pred = self.mask_head(mask_feats)  # body_feats

        self.cate_pred_list, self.kernel_pred_list = self.solov2_head(
            body_feats)

    # 在这里做文章
    def get_loss(self, ):
        loss = {}
        # get gt_ins_labels, gt_cate_labels, etc.
        gt_ins_labels, gt_cate_labels, gt_grid_orders = [], [], []
        fg_num = self.inputs['fg_num']
        for i in range(len(self.solov2_head.seg_num_grids)):
            ins_label = 'ins_label{}'.format(i)
            if ins_label in self.inputs:
                gt_ins_labels.append(self.inputs[ins_label])
            cate_label = 'cate_label{}'.format(i)
            if cate_label in self.inputs:
                gt_cate_labels.append(self.inputs[cate_label])
            grid_order = 'grid_order{}'.format(i)
            if grid_order in self.inputs:
                gt_grid_orders.append(self.inputs[grid_order])

        loss_solov2 = self.solov2_head.get_loss(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            gt_ins_labels, gt_cate_labels, gt_grid_orders, fg_num)
        loss.update(loss_solov2)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        seg_masks, cate_labels, cate_scores, bbox_num = self.solov2_head.get_prediction(
            self.cate_pred_list, self.kernel_pred_list, self.seg_pred,
            self.inputs['im_shape'], self.inputs['scale_factor'])
        outs = {
            "segm": seg_masks,
            "bbox_num": bbox_num,
            'cate_label': cate_labels,
            'cate_score': cate_scores
        }
        return outs
