import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")


mot_class_labels = ('person',)



class MOTDataset(Dataset):
    """
    MOT dataset class.
    """
    def __init__(self, 
                 img_size=640,
                 data_dir=None, 
                 image_set='train',
                 json_file='train_half.json',
                 transform=None):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            debug (bool): if True, only one data id is selected from the dataset
        """
        print('==============================')
        print('Image Set: {}'.format(image_set))
        print('Json file: {}'.format(json_file))
        print('==============================')
        self.img_size = img_size
        self.image_set = image_set
        self.json_file = json_file
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        # augmentation
        self.transform = transform


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target


    def load_image_target(self, img_id):
        # load an img info
        im_ann = self.coco.loadImgs(img_id)[0] 
        frame_id = im_ann["frame_id"]
        video_id = im_ann["video_id"]

        # annotation
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)

        # read image
        if "file_name" in im_ann: 
            file_name = im_ann["file_name"]
        else:
            file_name = "{:012}".format(img_id) + ".jpg"
        img_file = os.path.join(self.data_dir, self.image_set, file_name)
        image = cv2.imread(img_file)
        assert image is not None

        # image info
        height, width = image.shape[:2]
        img_info = (height, width, frame_id, video_id, file_name)
        
        #load a target
        objs = []
        for obj in annotations:
            if 'bbox' in obj:   
                x1 = np.max((0, obj['bbox'][0]))
                y1 = np.max((0, obj['bbox'][1]))
                x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    objs.append(obj)

        num_objs = len(objs)
        target = np.zeros((num_objs, 6))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            target[ix, 0:4] = obj["clean_bbox"]
            target[ix, 4] = cls
            target[ix, 5] = obj["track_id"]

        del im_ann, annotations

        return (image, target, img_info)


    def pull_item(self, index):
        # load an image and target
        img_id = self.ids[index]
        image, target, img_info = self.load_image_target(img_id)

        # preprocess
        image, target = self.transform(image, target)

        return image, target, img_info, np.array(img_id)


    def pull_image(self, index):
        img_id = self.ids[index]
        im_ann = self.coco.loadImgs(img_id)[0] 
        # read image
        if "file_name" in im_ann: 
            file_name = im_ann["file_name"]
        else:
            file_name = "{:012}".format(img_id) + ".jpg"
        img_file = os.path.join(self.data_dir, self.image_set, file_name)
        image = cv2.imread(img_file)
        assert image is not None

        return image, img_id


    def pull_anno(self, index):
        img_id = self.ids[index]
        # annotation
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)
        
        #load a target
        objs = []
        for obj in annotations:
            if 'bbox' in obj:   
                x1 = obj['bbox'][0]
                y1 = obj['bbox'][1]
                x2 = x1 + obj['bbox'][2]
                y2 = y1 + obj['bbox'][3]
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2, y2]
                    objs.append(obj)

        return objs


if __name__ == "__main__":
    from transforms import BaseTransform
    
    img_size = 640
    transform = BaseTransform(img_size=img_size,)

    dataset = MOTDataset(
        img_size=img_size,
        data_dir='/mnt/share/ssd2/dataset/MOT17/',
        image_set='train',
        json_file='val_half.json',
        transform=transform
        )
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target, img_info, img_id = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target[..., :4].numpy()
        labels = target[..., 4].numpy()
        track_id = target[..., 5].numpy()

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cls_id = int(label)
            # class name
            label = mot_class_labels[cls_id]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (x1, y1 - 5), 0, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)