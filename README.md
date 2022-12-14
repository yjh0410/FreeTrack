# FreeTrack

## Detector

[FreeYOLO](https://github.com/yjh0410/FreeYOLO).

## Tracker

[ByteTrack](https://github.com/ifzhang/ByteTrack)

## Demo
# Demo
* images tracking
```Shell
python demo.py --mode image \
               --path_to_img path/to/images/ \
               --detector yolo_free_tiny \
               --tracker byte_tracker \
               --img_size 640 \
               --cuda \
               --show
```

* video tracking

```Shell
python demo.py --mode video \
               --path_to_img path/to/video/ \
               --detector yolo_free_tiny \
               --tracker byte_tracker \
               --img_size 640 \
               --cuda \
               --show
```

* camera tracking

```Shell
python demo.py --mode camera \
               --detector yolo_free_tiny \
               --tracker byte_tracker \
               --img_size 640 \
               --cuda \
               --show
```
