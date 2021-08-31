#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2020. All rights reserved.
Created by C. L. Wang on 28.12.20
"""
import torch

from models.experimental import attempt_load
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class ImgDetector(object):
    """
    图像检测
    """
    def __init__(self, weights=None):
        if not weights:
            self.weights = os.path.join(DATA_DIR, 'models', 'best-3c-20210722.pt')
        else:
            self.weights = weights
        print('[Info] 模型路径: {}'.format(self.weights))

        self.img_size = 640
        self.conf_thres = 0.001
        self.iou_thres = 0.6

        self.device = select_device()  # 自动选择环境
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.img_size = self.load_model()  # 加载模型

    def load_model(self):
        """
        加载模型
        """
        # Load model
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        if self.half:
            model.half()
        model.eval()
        # if self.is_half:
        #     model.half()  # to FP16
        img_size = check_img_size(self.img_size, s=model.stride.max())  # check img_size

        # 设置Img Half
        img_tmp = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = model(img_tmp.half() if self.half else img_tmp) if self.device.type != 'cpu' else None  # run once

        return model, img_size

    def load_img(self, img_bgr):
        """
        加载图像
        """
        h0, w0 = img_bgr.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_bgr = cv2.resize(img_bgr, (int(w0 * r), int(h0 * r)),
                                 interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        return img_bgr, (h0, w0), img_bgr.shape[:2]  # img, hw_original, hw_resized

    def get_letter_shape(self, img_bgr, img_size=640, stride=32, pad=0.5):
        """
        获取填充的shape
        """
        h, w, _ = img_bgr.shape
        ar = h / w  # aspect ratio
        r_shape = [1, 1]
        if ar < 1:
            r_shape = [ar, 1]
        elif ar > 1:
            r_shape = [1, 1 / ar]
        letter_shape = np.ceil(np.array(r_shape) * img_size / stride + pad).astype(np.int) * stride
        return letter_shape

    def preprocess_data(self, img_bgr):
        """
        图像预处理
        """
        img, (h0, w0), (h, w) = self.load_img(img_bgr)
        letter_shape = self.get_letter_shape(img_bgr)

        img, ratio, pad = letterbox(img, letter_shape, auto=False, scaleup=False)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # 正则化
        img = torch.from_numpy(img)
        img = img.to(self.device, non_blocking=True)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, shapes

    @staticmethod
    def filter_clz_dict(clz_dict,  m_label="0"):
        """
        过滤不同类别框之间的重叠部分
        """
        clz_list = list(clz_dict.keys())

        # 没有主标签 或者 标签只有1个
        if len(clz_list) <= 1 or m_label not in clz_list:
            res_dict = clz_dict
        else:
            res_dict = collections.defaultdict(list)
            main_boxes = clz_dict[m_label]
            for clz in clz_list:
                if clz == m_label:
                    continue
                other_boxes = clz_dict[clz]
                flags = len(other_boxes) * [True]
                for idx, other_box in enumerate(other_boxes):
                    for main_box in main_boxes:
                        v_iou = min_iou(main_box, other_box)
                        if v_iou > 0.6:
                            flags[idx] = False
                            continue
                for idx, f in enumerate(flags):
                    if f:
                        res_dict[clz].append(other_boxes[idx])
            res_dict[m_label] = main_boxes

        # 过滤重叠框
        for idx, clz in enumerate(res_dict.keys()):
            box_list = res_dict[clz]
            box_list, _ = filer_boxes_by_size(box_list)
            res_dict[clz] = box_list

        return res_dict

    def detect_image(self, img_bgr):
        """
        图像检测逻辑
        """
        img, shapes = self.preprocess_data(img_bgr)  # 预处理数据
        out = self.model(img, augment=False)[0]  # 预测图像

        # NMS后处理
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, labels=[], multi_label=True, agnostic=False)

        clz_dict = collections.defaultdict(list)
        for si, pred in enumerate(out):
            predn = pred.clone()
            scale_coords(img.shape[2:], predn[:, :4], shapes[0], shapes[1])  # native-space pred
            predn = predn.tolist()
            for *xyxy, conf, cls in predn:  # 绘制图像
                xyxy = [int(i) for i in xyxy]
                conf = round(conf, 4)
                cls = str(int(cls))
                if conf > 0.2:
                    clz_dict[cls].append(xyxy)
        clz_dict = self.filter_clz_dict(clz_dict)  # 过滤
        return clz_dict


def process():
    img_path = os.path.join(DATA_DIR, 'images', '0900c7acdf107184d65956a785e55672.jpg')
    # img_path = os.path.join(DATA_DIR, 'images', '00458809b152dbd9d696da654fb7a2dd_2.jpg')
    # img_path = os.path.join(DATA_DIR, 'images', '009a3a4059168d1a6ace58a6ab536f66_2.jpg')
    img_bgr = cv2.imread(img_path)
    show_img_bgr(img_bgr)
    idet = ImgDetector()
    clz_dict = idet.detect_image(img_bgr)
    color_list = [(0, 0, 255), (0, 255, 0)]
    for idx, clz in enumerate(clz_dict.keys()):
        box_list = clz_dict[clz]
        box_list, _ = filer_boxes_by_size(box_list)
        img_bgr = draw_box_list(
            img_bgr, color=color_list[clz], is_new=False, box_list=box_list, save_name="tmp1.jpg")


def main():
    process()


if __name__ == '__main__':
    main()
