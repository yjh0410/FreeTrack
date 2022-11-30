# import tracker
from .tracker_free.build import build_tracker_free
# import detector
from .yolo_free.build import build_yolo_free


def build_tracker(args):
    if args.tracker == 'tracker_free':
        return build_tracker_free(args)
    else:
        raise NotImplementedError


def build_detector(args, cfg):
    if args.detector in ['yolo_free_nano', 'yolo_free_tiny',\
                         'yolo_free_large', 'yolo_free_huge']:
        return build_yolo_free(args, cfg, num_classes=args.num_classes)
    else:
        raise NotImplementedError
