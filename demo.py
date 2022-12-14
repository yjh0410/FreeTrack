import os
import cv2
import time
import argparse
import numpy as np
import torch

from dataset.transforms import BaseTransform
from utils.vis_tools import plot_tracking

from config import build_config

from models.build import build_tracker
from models.build import build_detector

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def parse_args():
    parser = argparse.ArgumentParser(description='FreeTrack Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    # data
    parser.add_argument('--mode', type=str, default='image',
                        help='image, video or camera')
    parser.add_argument('--path_to_img', type=str, default='dataset/demo/images/',
                        help='Dir to load images')
    parser.add_argument('--path_to_vid', type=str, default='dataset/demo/videos/',
                        help='Dir to load a video')
    parser.add_argument('--path_to_save', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('--fps', type=int, default=30,
                        help='frame rate')
    parser.add_argument('--show', action='store_true', default=False, 
                        help='show results.')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='save results.')

    # tracker
    parser.add_argument('--tracker', default='tracker_free', type=str,
                        help='build FreeTrack')
    parser.add_argument("--track_thresh", type=float, default=0.5, 
                        help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, 
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, 
                        help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which \
                              aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10,
                        help='filter out tiny boxes')
    parser.add_argument("--mot20", default=False, action="store_true",
                        help="test mot20.")

    # detector
    parser.add_argument('--detector', default='yolo_free_large', type=str,
                        help='build FreeYOLO')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='number of object classes.')

    # post process
    parser.add_argument('--conf_thresh',type=float, default=0.1,
                        help='conference threshold')
    parser.add_argument('--nms_thresh',type=float, default=0.5,
                        help='conference threshold')

    return parser.parse_args()


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def run(args,
        tracker,
        detector,
        post_processor, 
        device, 
        transform):
    save_path = os.path.join(args.path_to_save, args.mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if args.mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                # ------------------------- Detection ---------------------------
                # preprocess
                x, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)

                # detect
                t0 = time.time()
                outputs = detector(x)
                print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

                # post process
                t1 = time.time()
                bboxes, scores, labels = post_processor(outputs)
                bboxes /= ratio
                print("post-process time: {:.1f} ms".format((time.time() - t1)*1000))

                # track
                t2 = time.time()
                if len(bboxes) > 0:
                    online_targets = tracker.update(scores, bboxes, labels)
                    online_xywhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        xywh = t.xywh
                        tid = t.track_id
                        vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                        if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                            online_xywhs.append(xywh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
                    print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                    
                    # plot tracking results
                    online_im = plot_tracking(
                        frame, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                    )
                else:
                    online_im = frame

                # show results
                if args.show:
                    cv2.imshow('tracking', online_im)

            else:
                break
            frame_id += 1

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif args.mode == 'image':
        files = get_image_list(args.path_to_img)
        files.sort()
        # start tracking
        frame_id = 0
        results = []
        for frame_id, img_path in enumerate(files, 1):
            image = cv2.imread(os.path.join(img_path))
            # preprocess
            x, ratio = transform(image)
            x = x.unsqueeze(0).to(device)
            img_size = (x.shape[2], x.shape[3]) # (img_h, img_w)

            # detect
            t0 = time.time()
            outputs = detector(x)
            print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

            # post process
            t1 = time.time()
            bboxes, scores, labels = post_processor(img_size, outputs)
            bboxes /= ratio
            print("post-process time: {:.1f} ms".format((time.time() - t1)*1000))

            # track
            t2 = time.time()
            if len(bboxes) > 0:
                online_targets = tracker.update(scores, bboxes, labels)
                online_xywhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    xywh = t.xywh
                    tid = t.track_id
                    vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                    if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                        online_xywhs.append(xywh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                
                # plot tracking results
                online_im = plot_tracking(
                    image, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                )
            else:
                online_im = image

            # save results
            if args.save:
                vid_writer.write(online_im)
            # show results
            if args.show:
                cv2.imshow('tracking', online_im)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

            frame_id += 1

        cv2.destroyAllWindows()
            
    # ------------------------- Video ---------------------------
    elif args.mode == 'video':
        # read a video
        video = cv2.VideoCapture(args.path_to_vid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # path to save
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_path = os.path.join(save_path, timestamp, args.path.split("/")[-1])
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
        print("Save path: {}".format(save_path))

        # start tracking
        frame_id = 0
        results = []
        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                # preprocess
                x, ratio = transform(frame)
                x = x.unsqueeze(0).to(device)

                # detect
                t0 = time.time()
                outputs = detector(x)
                print("detect time: {:.1f} ms".format((time.time() - t0)*1000))

                # post process
                t1 = time.time()
                bboxes, scores, labels = post_processor(outputs)
                bboxes /= ratio
                print("post-process time: {:.1f} ms".format((time.time() - t1)*1000))

                # track
                t2 = time.time()
                if len(bboxes) > 0:
                    online_targets = tracker.update(scores, bboxes, labels)
                    online_xywhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        xywh = t.xywh
                        tid = t.track_id
                        vertical = xywh[2] / xywh[3] > args.aspect_ratio_thresh
                        if xywh[2] * xywh[3] > args.min_box_area and not vertical:
                            online_xywhs.append(xywh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{xywh[0]:.2f},{xywh[1]:.2f},{xywh[2]:.2f},{xywh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
                    print("tracking time: {:.1f} ms".format((time.time() - t2)*1000))
                    
                    # plot tracking results
                    online_im = plot_tracking(
                        frame, online_xywhs, online_ids, frame_id=frame_id + 1, fps=1. / (time.time() - t0)
                    )
                else:
                    online_im = frame

                # save results
                if args.save:
                    vid_writer.write(online_im)
                # show results
                if args.show:
                    cv2.imshow('tracking', online_im)

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        video.release()
        vid_writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(0)

    # config
    cfg = build_config(args)

    # build detector
    detector, post_processor = build_detector(args, cfg)
    detector = detector.to(device).eval()
    
    # build tracker
    tracker = build_tracker(args)

    # build post-processor
    # transform
    transform = BaseTransform(img_size=args.img_size)

    # run
    run(args=args,
        tracker=tracker,
        detector=detector, 
        device=device,
        post_processor=post_processor,
        transform=transform)
