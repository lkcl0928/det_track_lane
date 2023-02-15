from utils.datasets import *
from utils.plots import *
from utils.general import *
from utils.augmentations import *
from utils.torch_utils import load_classifier,select_device
from models.experimental import *
from detect import Detect
import torch 
import argparse
import os.path as osp
import glob
from tqdm import tqdm
from pathlib import Path
import os
from clrnet.utils.config import Config

@torch.no_grad()
class YOLO(object):
    def __init__(self, weights):
        set_logging()
        self.device = select_device(device="cuda:0")
        self.w = str(weights[0] if isinstance(weights, list) else weights)
        self.model = torch.jit.load(self.w) if 'torchscript' in self.w else attempt_load(weights, map_location=self.device)
        half=True
        if half:
            self.model.half()
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.color=[(0,255,0),(255,0,0),(0,0,255)]

    def detect_image(self,img):
        im0 = img.copy()
        stride= 32
        img = letterbox(im0, [640,640], stride, True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img  = img/255  # 0 - 255 to 0.0 - 1.0
        img = torch.from_numpy(img)
        img = torch.unsqueeze(img, dim=0).to(self.device)
        img = img.type(torch.cuda.HalfTensor)
        pred = self.model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, 0.65, 0.5)
        annotator = Annotator(im0, line_width=3, example=str(self.names))
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    c=int(cls)
                    label =  f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=self.color[c])
        im0 = annotator.result()
        return im0


def process_video(args):
    cfg = Config.fromfile(args.Configfile)
    cfg.show = args.show
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    yolo=YOLO(args.yolo_weights)
    cap=cv2.VideoCapture(args.video_path)
    parent, filename = os.path.split(args.video_path)
    #videoWidth, videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        #cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        im=yolo.detect_image(img)
        t1=time.time()
        im0=cv2.resize(im,(1640,590))
        im0s=detect.run(im0)
        cv2.putText(im0s, "FPS:%d" % (1/(t1-t0)), (40, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
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
    parser.add_argument('--video_path', type=str, default='/root/yolov5/CLRNet/lane.mp4', help='The path of input_video')
    parser.add_argument('--folder_path', default="/root/yolov5/CLRNet/detect" ,help='The path of the image_folder')
    args = parser.parse_args()
    process_video(args)
    #process_image(args)
    #process_folder(args)



