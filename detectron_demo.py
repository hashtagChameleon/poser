#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import numpy as np
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

model_path = '/home/levishai_g/pose_estimation/models/detectron-model.pkl'

class Detectron:
    def __init__(self):
        # workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.GlobalInit(['caffe2'])
        setup_logging(__name__)
        logger = logging.getLogger(__name__)
        merge_cfg_from_file('Detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml')
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)

        assert not cfg.MODEL.RPN_ONLY, \
            'RPN models are not supported'
        assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
            'Models that require precomputed proposals are not supported'

        self.model = infer_engine.initialize_model_from_cfg(model_path)
        self.thresh = 0.7

    def process_image(self, img):
        timers = defaultdict(Timer)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, img, None, timers=timers
            )
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = self.convert_from_cls_format(
                cls_boxes, cls_segms, cls_keyps)
        if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.thresh) and not out_when_no_box:
            return []

        if boxes is None:
            sorted_inds = [] # avoid crash when 'boxes' is None
        else:
            # Display in largest to smallest order to reduce occlusion
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_inds = np.argsort(-areas)

        mask_color_id = 0
        bbox_list = []
        for i in sorted_inds:
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.thresh:
                continue

            if classes[i] == 1:
                xmin = bbox[0]
                ymin = bbox[1]
                width = bbox[2] - xmin
                height = bbox[3] - ymin
                bbox_list.append([xmin, ymin, width, height])
                # print(f'raven: [{xmin} {ymin} {width} {height}] score {round(score, 2)}')
        return bbox_list

    def convert_from_cls_format(self, cls_boxes, cls_segms, cls_keyps):
        """Convert from the class boxes/segms/keyps format generated by the testing
        code.
        """
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        if cls_segms is not None:
            segms = [s for slist in cls_segms for s in slist]
        else:
            segms = None
        if cls_keyps is not None:
            keyps = [k for klist in cls_keyps for k in klist]
        else:
            keyps = None
        classes = []
        for j in range(len(cls_boxes)):
            classes += [j] * len(cls_boxes[j])
        return boxes, segms, keyps, classes
