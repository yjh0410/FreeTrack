# yolo config


yolo_config = {
    'yolo_free_nano': {
        # model
        'backbone': 'shufflenetv2_1.0x',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'sppf',
        'neck_dim': 232,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'nano',
        'fpn_dim': [116, 232, 232],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        },

    'yolo_free_tiny': {
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny', # 'tiny', 'large', 'huge
        'fpn_dim': [128, 256, 256],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        },

    'yolo_free_large': {
        # model
        'backbone': 'elannet_large',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 512,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'large', # 'tiny', 'large', 'huge
        'fpn_dim': [512, 1024, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        },

    'yolo_free_huge': {
        # model
        'backbone': 'elannet_huge',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 640,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'huge', # 'tiny', 'large', 'huge
        'fpn_dim': [640, 1280, 640],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 320,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.1,
        'nms_thresh': 0.5,
        },

}