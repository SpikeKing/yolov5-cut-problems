#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 8.7.21
"""

import os
import sys
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.make_html_page import make_html_page
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class DataProcessor(object):
    """
    英语单词数据处理
    """
    def __init__(self):
        self.file_name = os.path.join(DATA_DIR, 'files', 'dump_train_kousuan-20210906.txt')
        self.out_dir = os.path.join(DATA_DIR, 'dataset_cut_problems_20210906')
        mkdir_if_not_exist(self.out_dir)
        self.imgs_dir = os.path.join(self.out_dir, 'images')
        self.lbls_dir = os.path.join(self.out_dir, 'labels')
        mkdir_if_not_exist(self.imgs_dir)
        mkdir_if_not_exist(self.lbls_dir)
        self.train_imgs_dir = os.path.join(self.imgs_dir, 'train')
        self.val_imgs_dir = os.path.join(self.imgs_dir, 'val')
        mkdir_if_not_exist(self.train_imgs_dir)
        mkdir_if_not_exist(self.val_imgs_dir)
        self.train_lbls_dir = os.path.join(self.lbls_dir, 'train')
        self.val_lbls_dir = os.path.join(self.lbls_dir, 'val')
        mkdir_if_not_exist(self.train_lbls_dir)
        mkdir_if_not_exist(self.val_lbls_dir)

        time_str = get_current_time_str()
        self.out_txt = os.path.join(self.out_dir, "check-{}.txt".format(time_str))
        self.out_html = os.path.join(self.out_dir, "check-{}.html".format(time_str))

    @staticmethod
    def convert(iw, ih, box):
        """
        将标注的xml文件标注转换为darknet形的坐标
        """
        iw = float(iw)
        ih = float(ih)
        dw = 1. / iw
        dh = 1. / ih
        x = (box[0] + box[2]) / 2.0 - 1
        y = (box[1] + box[3]) / 2.0 - 1
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    @staticmethod
    def save_img_path(img_bgr, img_name, oss_root_dir=""):
        """
        上传图像
        """
        from x_utils.oss_utils import save_img_2_oss
        if not oss_root_dir:
            oss_root_dir = "zhengsheng.wcl/problems_segmentation/kousuan/imgs-tmp/{}".format(get_current_day_str())
        img_url = save_img_2_oss(img_bgr, img_name, oss_root_dir)
        return img_url

    @staticmethod
    def process_line(idx, data_line, imgs_dir, lbls_dir, out_txt):
        data_line = data_line.replace("\'", "\"")
        data_dict = json.loads(data_line)
        img_url = data_dict['url']
        point_boxes = data_dict['coord']

        img_name = img_url.split("/")[-1].split(".")[0]

        # 不同文件使用不同的文件名
        file_idx = str(idx).zfill(5)
        img_path = os.path.join(imgs_dir, 'v1_{}_{}.jpg'.format(file_idx, img_name))
        lbl_path = os.path.join(lbls_dir, 'v1_{}_{}.txt'.format(file_idx, img_name))

        # 写入图像
        is_ok, img_bgr = download_url_img(img_url)
        cv2.imwrite(img_path, img_bgr)  # 写入图像

        # 写入标签
        ih, iw, _ = img_bgr.shape  # 高和宽
        res_bboxes_lines = []

        bbox_list = []
        for point_bbox in point_boxes:
            bbox_list.append(rec2bbox(point_bbox))
        # draw_box_list(img_bgr, bbox_list, is_show=True)
        img_out = draw_rec_list(img_bgr, point_boxes)
        img_out_url = DataProcessor.save_img_path(img_out, "{}-{}.jpg".format(img_name, get_current_time_str()))
        write_line(out_txt, "{}\t{}".format(img_out_url, img_name))

        # 写入3个不同标签
        for bbox in bbox_list:
            bbox_yolo = DataProcessor.convert(iw, ih, bbox)
            bbox_yolo = [str(round(i, 6)) for i in bbox_yolo]
            res_bboxes_lines.append(" ".join(["0", *bbox_yolo]))

        create_file(lbl_path)
        write_list_to_file(lbl_path, res_bboxes_lines)
        print('[Info] idx: {} 处理完成: {}'.format(idx, img_path))

    def process(self):
        print('[Info] 处理数据: {}'.format(self.file_name))
        data_lines = read_file(self.file_name)
        data_lines = data_lines[:50]
        n_lines = len(data_lines)
        random.seed(47)
        random.shuffle(data_lines)
        print('[Info] 文件数: {}'.format(n_lines))

        n_x = 20
        n_split = len(data_lines) // n_x
        train_lines = data_lines[:n_split*(n_x-1)]
        val_lines = data_lines[n_split*(n_x-1):]
        print('[Info] 训练: {}, 测试: {}'.format(len(train_lines), len(val_lines)))

        pool = Pool(processes=100)

        for idx, data_line in enumerate(train_lines):
            # DataProcessor.process_line(idx, data_line, self.train_imgs_dir, self.train_lbls_dir, self.out_txt)
            pool.apply_async(DataProcessor.process_line,
                             (idx, data_line, self.train_imgs_dir, self.train_lbls_dir, self.out_txt))

        for idx, data_line in enumerate(val_lines):
            # DataProcessor.process_line(idx, data_line, self.val_imgs_dir, self.val_lbls_dir, self.out_txt)
            pool.apply_async(DataProcessor.process_line,
                             (idx, data_line, self.val_imgs_dir, self.val_lbls_dir, self.out_txt))

        pool.close()
        pool.join()
        print('[Info] 处理完成: {}'.format(self.out_dir))

        data_lines = read_file(self.out_txt)
        out_list = []
        for data_line in data_lines:
            items = data_line.split("\t")
            out_list.append(items)
        make_html_page(self.out_html, out_list)
        print('[Info] 写入完成: {}'.format(self.out_html))


def main():
    dp = DataProcessor()
    dp.process()


if __name__ == '__main__':
    main()
