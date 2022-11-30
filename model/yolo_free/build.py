#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolo_free import FreeYOLO
from .post_process import PostProcessor


# build object detector
def build_model(args, cfg, device, num_classes=1, weight=None):    
    # model
    model = FreeYOLO(cfg=cfg, device=device, num_classes=num_classes,)

    # Load COCO pretrained weight
    assert weight is not None, \
        print('You shoule load the trained ckpt file for FreeYOLO.')
    checkpoint_state_dict = torch.load(weight, map_location='cpu').pop("model")
    model.load_state_dict(checkpoint_state_dict)

    # post process
    post_processor = PostProcessor(
        img_size=args.img_size,
        strides=model.stride,
        num_classes=num_classes,
        conf_thresh=cfg['conf_thresh'],
        nms_thresh=cfg['nms_thresh'],
    )

    return model, post_processor
