import numpy as np


def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # nms
    keep = nms(bboxes, scores, nms_thresh)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep = np.zeros(len(bboxes), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_classes)


# Post-Process
class PostProcessor(object):
    def __init__(self, img_size, strides, num_classes, conf_thresh=0.1, nms_thresh=0.5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = strides
        self.img_size = None


    def generate_anchors(self, img_size):
        """
            fmp_size: (List) [H, W]
        """
        if img_size == self.img_size:
            return self.anchors, self.expand_strides
        else:
            all_anchors = []
            all_expand_strides = []
            img_h, img_w = img_size
            for stride in self.strides:
                # generate grid cells
                fmp_h, fmp_w = img_h // stride, img_w // stride
                anchor_x, anchor_y = np.meshgrid(np.arange(fmp_w), np.arange(fmp_h))
                # [H, W, 2]
                anchor_xy = np.stack([anchor_x, anchor_y], axis=-1)
                shape = anchor_xy.shape[:2]
                # [H, W, 2] -> [HW, 2]
                anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
                all_anchors.append(anchor_xy)

                # expanded stride
                strides = np.full((*shape, 1), stride)
                all_expand_strides.append(strides.reshape(-1, 1))

            anchors = np.concatenate(all_anchors, axis=0)
            expand_strides = np.concatenate(all_expand_strides, axis=0)

            self.anchors = anchors
            self.expand_strides = expand_strides

            return anchors, expand_strides                


    def decode_boxes(self, anchors, strides, pred_regs):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors[..., :2] + pred_regs[..., :2] * strides
        # size of bbox
        pred_box_wh = np.exp(pred_regs[..., 2:]) * strides

        # cwcywh -> x1y1x2y2
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

        return pred_box


    def __call__(self, img_size, predictions):
        """
        Input:
            img_size: (Tulpe): [img_h, img_w]
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        reg_preds = predictions[..., :4]
        obj_preds = predictions[..., 4:5]
        cls_preds = predictions[..., 5:]
        scores = np.sqrt(obj_preds * cls_preds)

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # generate anchors
        anchors, strides = self.generate_anchors(img_size)
        
        # bboxes
        bboxes = self.decode_boxes(anchors, strides, reg_preds)     # [M, 4]    

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        return bboxes, scores, labels
