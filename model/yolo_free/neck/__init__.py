from .spp import SPPBlockCSP, SPPF
from .pafpn import PaFPNELAN


def build_fpn(cfg, in_dims, out_dim):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'pafpn_elan':
        fpn_net = PaFPNELAN(in_dims=in_dims,
                            out_dim=out_dim,
                            fpn_size=cfg['fpn_size'],
                            depthwise=cfg['fpn_depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])
                                                        
    return fpn_net


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'spp_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    elif model == 'sppf':
        neck = SPPF(in_dim, out_dim, k=cfg['pooling_size'])

    return neck
    