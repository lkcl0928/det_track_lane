from utils.datasets import *
from utils.plots import *
from utils.general import *
from utils.augmentations import *
from utils.torch_utils import load_classifier,select_device
from models.experimental import *
from det import Detect
import torch 
import argparse
import os.path as osp
import glob
from tqdm import tqdm
from pathlib import Path
import os
from clrnet.utils.config import Config


def process_video(args):
    cfg = Config.fromfile(args.Configfile)
    cfg.show = args.show
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    yolo=YOLO(args.yolo_weights)
    cap=cv2.VideoCapture(args.video_path)
    parent, filename = os.path.split(args.video_path)
    videoWidth, videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frameAll = cap.get(7)
    idx_frame = 0
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #size = (videoWidth, videoHeight)
    if os.path.exists(args.savedir):
        shutil.rmtree(args.savedir)
    os.mkdir(args.savedir)
    vout = cv2.VideoWriter(args.savedir+"processed.mp4", fourcc, fps, (1640,590))
    while cap.isOpened():
        idx_frame +=1
        print('frame%d frameAll:%d' % (idx_frame, frameAll))
        ret,img=cap.read()
        t0=time.time()
        t1=time.time()
        im0=cv2.resize(img,(1640,590))
        im0s=detect.run(im0)
        if args.show:
            cv2.imshow(str(filename),im0s)
            cv2.waitkey(1)
        vout.write(im0s)
        if idx_frame==frameAll:
            break
    cap.release()
    vout.release()
    #cv2.destroyAllWindows()

def process_image(args):
    cfg = Config.fromfile(args.Configfile)
    cfg.show = args.show
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    yolo=YOLO(args.yolo_weights)
    parent, filename = os.path.split(args.img_path)
    img=cv2.imread(args.img_path)
    im=yolo.detect_image(img)
    im0=cv2.resize(im,(1640,590))
    im0s=detect.run(im0)
    if os.path.exists(args.savedir):
        shutil.rmtree(args.savedir)
    os.mkdir(args.savedir)
    if args.show:
        cv2.imshow(str(filename),im0s)
        cv2.waitkey(0)
    cv2.imwrite(args.savedir+filename,im0s)
    print('%s has saved'%(filename))
    #cv2.destroyAllWindows()


def process_folder(args):
    cfg = Config.fromfile(args.Configfile)
    cfg.show = args.show
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    yolo=YOLO(args.yolo_weights)
    files=os.listdir(args.folder_path)
    if os.path.exists(args.savedir):
        shutil.rmtree(args.savedir)
    os.mkdir(args.savedir)
    for file in files:
        file_path=os.path.join(args.folder_path,file)
        parent, filename = os.path.split(file_path)
        img=cv2.imread(file_path)
        im=yolo.detect_image(img)
        im0=cv2.resize(im,(1640,590))
        im0s=detect.run(im0)
        if args.show:
            cv2.imshow(str(filename),im0s)
            cv2.waitkey(1)
        cv2.imwrite(args.savedir+filename,im0s)
        print('%s has saved'%(filename))
    #cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Configfile',default='/root/yolov5/CLRNet/configs/clrnet/clr_resnet34_culane.py',help='config file path')
    parser.add_argument('--img_path', default="/root/yolov5/CLRNet/1195.jpg" ,help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show',default=False, action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default='/root/yolov5/CLRNet/detec/', help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='/root/yolov5/CLRNet/culane_r34.pth', help='The path of model')
    parser.add_argument('--yolo_weights', type=str, default='/root/yolov5/best.pt', help='The path of model')
    parser.add_argument('--video_path', type=str, default='/root/yolov5/CLRNet/processed.mp4', help='The path of input_video')
    parser.add_argument('--folder_path', default="/root/yolov5/CLRNet/detect" ,help='The path of the image_folder')
    args = parser.parse_args()
    process_video(args)
    #process_image(args)
    #process_folder(args)



