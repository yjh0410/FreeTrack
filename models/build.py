# import tracker
from .byte_tracker.build import build_byte_tracker
# import detector
from .yolo_free.build import build_yolo_free


def build_tracker(args):
    if args.tracker == 'byte_tracker':
        return build_byte_tracker(args)
    else:
        raise NotImplementedError


def build_detector(args, cfg):
    if args.detector in ['yolo_free_nano', 'yolo_free_tiny',\
                         'yolo_free_large', 'yolo_free_huge']:
        return build_yolo_free(args, cfg, num_classes=args.num_classes)
    else:
        raise NotImplementedError
