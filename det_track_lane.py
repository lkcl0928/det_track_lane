import warnings
from yolov5 import YOLO
from deep_sort import build_tracker
from utils.draw import pil_draw_box
from utils.plots import *
from utils.general import *
from utils.parser import get_config
from lanedet import Detect
from pathlib import Path
import os
import argparse
import shutil
from clrnet.utils.config import Config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoTracker(object):
    def __init__(self):
        warnings.filterwarnings("ignore")
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)
        self.detector = YOLO()
        cfg = get_config()
        cfg.merge_from_file('./configs/deep_sort.yaml')
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.vdo = cv2.VideoCapture()
        self.frameAll = 0
        self.idx_frame = 0
        self.dict_box = {}

    def run(self, args):
        global x_center, y_center
        cfg = Config.fromfile(args.Configfile)
        cfg.show = args.show
        cfg.load_from = args.load_from
        detect = Detect(cfg)
        parent, filename = os.path.split(args.video_path)
        self.vdo = cv2.VideoCapture(args.video_path)
        self.frameAll = self.vdo.get(7)
        self.idx_frame = 0
        fps = self.vdo.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if os.path.exists(args.savedir):
            shutil.rmtree(args.savedir)
        os.mkdir(args.savedir)
        vout = cv2.VideoWriter(args.savedir+"processed.mp4", fourcc, fps, (1640,590))
        while self.vdo.isOpened():
            self.idx_frame += 1
            ret, img = self.vdo.read()
            if ret == True:
                im0 = img.copy()
                image = im0
                bbox_xywh, cls_conf, cls_ids, img = self.detector.detect_image(img)
                outputs = self.deepsort.update(bbox_xywh, cls_conf, image)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    font = ImageFont.truetype(font='/root/.config/Ultralytics/Arial.ttf',
                                              size=np.floor(0.012 * np.shape(img)[1]).astype('int32'))
                    img = pil_draw_box(img, bbox_xyxy, identities, font)
                imgg=cv2.resize(img,(1640,590))
                im0s=detect.run(imgg)
                #cv2.imshow(str(video_path), img)
                vout.write(im0s)
                print('frame%d frameAll:%d' % (self.idx_frame, self.frameAll))
                if self.idx_frame==20:
                    break
            else:
                break
        self.vdo.release()
        vout.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Configfile',default='/root/work/configs/clrnet/clr_resnet34_culane.py',help='config file path')
    parser.add_argument('--show',default=False, action='store_true', help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default='/root/work/result', help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='/root/work/culane_r34.pth', help='The path of model')
    parser.add_argument('--video_path', type=str, default='your video path', help='The path of input_video')
    args = parser.parse_args()
    vdo=VideoTracker()
    vdo.run(args)
