import torch
import torch.nn as nn

from .backbone import build_backbone
from .neck import build_neck, build_fpn
from .head import build_head


# Anchor-free YOLO
class FreeYOLO(nn.Module):
    def __init__(self, cfg, device, num_classes=1):
        super(FreeYOLO, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        
        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=bk_dim[-1], out_dim=cfg['neck_dim'])
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=cfg['fpn_dim'], out_dim=cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg) 
            for _ in range(len(cfg['stride']))
            ])

        ## pred
        head_dim = cfg['head_dim']
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, 1, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, self.num_classes, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ])                 

        # --------- Network Initialization ----------
        # init bias
        self.init_yolo()


    def init_yolo(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        for obj_pred in self.obj_preds:
            b = obj_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    @torch.no_grad()
    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # neck
        feats['layer4'] = self.neck(feats['layer4'])

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_obj_preds = []
        all_cls_preds = []
        all_reg_preds = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # [1, C, H, W] -> [H, W, C] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)

        # no decode in inference
        # the pipelline of post process is outside
        obj_preds = torch.cat(all_obj_preds, dim=0)
        cls_preds = torch.cat(all_cls_preds, dim=0)
        reg_preds = torch.cat(all_reg_preds, dim=0)

        # [n_anchors_all, 4 + 1 + C]
        outputs = torch.cat([reg_preds, obj_preds.sigmoid(), cls_preds.sigmoid()], dim=-1)

        return outputs
