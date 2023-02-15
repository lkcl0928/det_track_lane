import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
import shutil
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img):
        ori_img = img
        img0 = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img0, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'ori_img':ori_img})
        return data
    
    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            data = self.net.module.heads.get_lanes(data)
        return data

    def show(self, data):
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        img=imshow_lanes(data['ori_img'], lanes, show=self.cfg.show)
        return img

    def run(self, img):
        data = self.preprocess(img)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.load_from:
            im0=self.show(data)
        return im0


def process(args):
    cfg = Config.fromfile("/root/CLRNet/configs/clrnet/clr_resnet18_culane.py")
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    img=cv2.imread(args.img_path)
    detect.run(img)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', default="/root/CLRNet/2.jpg" ,help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show',default=False, action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='/root/CLRNet/culane_r18.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
    
