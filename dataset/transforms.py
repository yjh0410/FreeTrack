import cv2
import torch


# BaseTransform
class BaseTransform(object):
    def __init__(self, img_size=640):
        self.img_size =img_size


    def __call__(self, image):
        # resize
        img_h0, img_w0 = image.shape[:2]

        ratio = self.img_size / max(img_h0, img_w0)
        if ratio != 1: 
            img = cv2.resize(image, (int(img_w0 * ratio), int(img_h0 * ratio)))
        else:
            img = image

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        if img_h0 > img_w0:
            pad_img_h = self.img_size
            pad_img_w = (img_w0 // 32 + 1) * 32
        elif img_h0 < img_w0:
            pad_img_h = (img_h0 // 32 + 1) * 32
            pad_img_w = self.img_size
        else:
            pad_img_h = self.img_size
            pad_img_w = self.img_size
        pad_image = torch.ones([img_tensor.size(0), pad_img_h, pad_img_w]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor

        return pad_image, ratio

