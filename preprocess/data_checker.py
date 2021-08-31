#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2021. All rights reserved.
Created by C. L. Wang on 31.8.21
"""
import os
import sys
import argparse
from multiprocessing.pool import Pool

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from myutils.make_html_page import make_html_page
from myutils.cv_utils import *
from myutils.project_utils import *
from root_dir import DATA_DIR


class DataChecker(object):
    def __init__(self, in_folder, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder
        print('[Info] 输入文件夹: {}'.format(self.in_folder))
        print('[Info] 输出文件夹: {}'.format(self.out_folder))

    @staticmethod
    def check_darknet_data(ih, iw, data_line):
        items = data_line.split(" ")
        items = [float(i) for i in items]
        x, y, w, h = items[1:]
        x, y, w, h = x * iw, y * ih, w * iw, h * ih
        x_min, x_max = x - w // 2, x + w // 2
        y_min, y_max = y - h // 2, y + h // 2
        bbox = [int(x) for x in [x_min, y_min, x_max, y_max]]
        return bbox

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
    def process_item(img_idx, name, path, label_folder, out_txt):
        label_name = name.split('.')[0] + ".txt"
        img_bgr = cv2.imread(path)
        label_path = os.path.join(label_folder, label_name)
        data_lines = read_file(label_path)
        if len(data_lines) == 0:
            return
        bbox_list = []
        for idx, data_line in enumerate(data_lines):
            ih, iw, _ = img_bgr.shape
            bbox = DataChecker.check_darknet_data(ih, iw, data_line)
            bbox_list.append(bbox)

        img_out = draw_box_list(img_bgr, bbox_list)
        img_name = "{}-{}.jpg".format(name.split('.')[0], get_current_time_str())
        img_out_url = DataChecker.save_img_path(img_out, img_name)
        write_line(out_txt, "{}\t{}".format(img_out_url, name.split('.')[0]))
        print("[Info] 处理完成: {}".format(img_idx))

    def process(self):
        image_folder = os.path.join(self.in_folder, "images", "train")
        label_folder = os.path.join(self.in_folder, "labels", "train")
        paths_list, names_list = traverse_dir_files(image_folder)

        time_str = get_current_time_str()
        out_txt = os.path.join(self.out_folder, "check-{}.txt".format(time_str))
        out_html = os.path.join(self.out_folder, "check-{}.html".format(time_str))

        paths_list, names_list = shuffle_two_list(paths_list, names_list )
        paths_list, names_list = paths_list[:20], names_list[:20]
        print('[Info] 检查样本数: {}'.format(len(paths_list)))

        pool = Pool(processes=100)
        for img_idx, (path, name) in enumerate(zip(paths_list, names_list)):
            # DataChecker.process_item(img_idx, name, path, label_folder, out_txt)
            pool.apply_async(DataChecker.process_item, (img_idx, name, path, label_folder, out_txt))
        pool.close()
        pool.join()
        print('[Info] 检查完成! {}'.format(self.in_folder))

        data_lines = read_file(out_txt)
        out_list = []
        for data_line in data_lines:
            items = data_line.split("\t")
            out_list.append(items)
        make_html_page(out_html, out_list)
        print('[Info] 写入完成: {}'.format(out_html))


def parse_args():
    """
    处理脚本参数，支持相对路径
    """
    parser = argparse.ArgumentParser(description='服务测试')
    parser.add_argument('-i', dest='in_folder', required=False, help='测试文件夹', type=str)
    parser.add_argument('-o', dest='out_folder', required=False, help='输出文件夹', type=str)

    args = parser.parse_args()

    arg_in_folder = args.in_folder
    print("测试文件夹: {}".format(arg_in_folder))

    arg_out_folder = args.out_folder
    print("输出文件夹: {}".format(arg_out_folder))
    mkdir_if_not_exist(arg_out_folder)

    return arg_in_folder, arg_out_folder


def main():
    arg_in_folder, arg_out_folder = parse_args()
    dc = DataChecker(arg_in_folder, arg_out_folder)
    dc.process()


if __name__ == '__main__':
    main()
