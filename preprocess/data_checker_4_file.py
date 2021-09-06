#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 6.9.21
"""

import json
import os
import sys

import cv2
from multiprocessing.pool import Pool

from myutils.make_html_page import make_html_page

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class DataChecker4File(object):
    def __init__(self):
        self.old_file_name = os.path.join(DATA_DIR, "files", "dump_train_kousuan.txt")
        self.new_file_name = os.path.join(DATA_DIR, "files", "dump_train_kousuan-20210906.txt")
        self.out_imgs_url = os.path.join(DATA_DIR, "files", "dump_train_kousuan-20210906.res.txt")
        self.out_imgs_html = os.path.join(DATA_DIR, "files", "dump_train_kousuan-20210906.res.html")

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
    def process_line(data_idx, data_line, out_file):
        data_line = data_line.replace("\'", "\"")
        data_dict = json.loads(data_line)
        img_url = data_dict['url']
        point_boxes = data_dict['coord']
        is_ok, img_bgr = download_url_img(img_url)
        img_out = draw_rec_list(img_bgr, point_boxes)

        img_name = img_url.split("/")[-1].split(".")[0]
        out_img_name = "{}-{}-{}.jpg".format(data_idx, img_name, get_current_time_str())
        out_img_url = DataChecker4File.save_img_path(img_out, out_img_name)
        write_line(out_file, out_img_url)
        print('[Info] 处理完成!')

    def process(self):
        old_data_lines = read_file(self.old_file_name)
        new_data_lines = read_file(self.new_file_name)
        print('[Info] old_file_name: {}, 样本数: {}'.format(self.old_file_name, len(old_data_lines)))
        print('[Info] new_file_name: {}, 样本数: {}'.format(self.new_file_name, len(new_data_lines)))

        pool = Pool(processes=100)
        for data_idx, new_data_line in enumerate(new_data_lines):
            # DataChecker4File.save_img_path(data_idx, new_data_line, self.out_imgs_url)
            pool.apply_async(DataChecker4File.save_img_path, (data_idx, new_data_line, self.out_imgs_url))
        pool.close()
        pool.join()

        print('[Info] 处理完成 {}'.format(self.out_imgs_url))
        data_lines = read_file(self.out_imgs_url)
        out_list = []
        for data_line in data_lines:
            out_list.append(data_line)
        make_html_page(self.out_imgs_html, out_list)
        print('[Info] 写入完成: {}'.format(self.out_imgs_html))


def main():
    dc4f = DataChecker4File()
    dc4f.process()


if __name__ == "__main__":
    main()
