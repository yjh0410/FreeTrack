#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.hub import load_state_dict_from_url

from .yolo_free import FreeYOLO
from .post_process import PostProcessor


model_urls = {
    'yolo_free_nano': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth',
    'yolo_free_tiny': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth',
    'yolo_free_large': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_ch.pth',
    'yolo_free_huge': 'https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_huge_ch.pth',
}


# build object detector
def build_yolo_free(args, cfg, num_classes=1):    
    # model
    model = FreeYOLO(cfg=cfg, num_classes=num_classes,)

    # Load COCO pretrained weight
    url = model_urls[args.detector]

    # check
    assert url is not None, \
        print('No detector weight for {}...'.format(args.detector))


    # state dict
    print('Loading detector weight ...')
    checkpoint = load_state_dict_from_url(url, map_location='cpu')
    checkpoint_state_dict = checkpoint.pop('model')
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
