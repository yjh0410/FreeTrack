from .tracker_free import FreeTracker


def build_tracker_free(args):
    tracker = FreeTracker(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        frame_rate=args.fps,
        match_thresh=args.match_thresh,
        mot20=args.mot20
    )

    return tracker
    