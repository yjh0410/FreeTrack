from .elannet import build_elannet
from .shufflenetv2 import build_shufflenetv2


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if cfg['backbone'] in ['elannet_large', 'elannet_huge', \
                           'elannet_tiny', 'elannet_nano']:
        model, feat_dim = build_elannet(model_name=cfg['backbone'])
    
    elif cfg['backbone'] in ['shufflenetv2_0.5x', 'shufflenetv2_1.0x']:
        model, feat_dim = build_shufflenetv2(model_size=cfg['backbone'][-4:])
        
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
