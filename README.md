# FreeTrack

* Detector

I use my [FreeYOLO](https://github.com/yjh0410/FreeYOLO) as the detector, which is only pretrained on CrowdHuman dataset.

| Model          |  Scale  |    AP    |    AP50    |  Weight  |
|----------------|---------|----------|------------|----------|
| FreeYOLO-Nano  |  640    |   31.3   |   67.2     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_nano_ch.pth) |
| FreeYOLO-Tiny  |  640    |   34.7   |   70.4     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_tiny_ch.pth) |
| FreeYOLO-Large |  640    |   43.1   |   76.5     | [github](https://github.com/yjh0410/FreeYOLO/releases/download/weight/yolo_free_large_ch.pth) |
| FreeYOLO-Huge  |  640    |      |        |  |

* Tracker

I use the simple and strong [ByteTrack](https://github.com/ifzhang/ByteTrack) as the tracker.

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
