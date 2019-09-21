from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.test_demo import hoi_detect

from torchvision.ops import nms

from utils.timer import Timer
import numpy as np
import os, cv2
import argparse
import glob
import pickle
import json
import threading
import time
import datetime
import sys

from nets.resnet_v1 import resnetv1
from nets.iCAN_ResNet_VCOCO import resnet18
from visualization import vis_detection

import torch
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Demo hoi_detection')
    parser.add_argument('--demo', dest='demo_file',
            help='the address of the demo file', default=None, type=str, required=True)
    parser.add_argument('-t', '--type', dest='type',
            help='the type of the demo file, could be "image", "video", "camera" or "time", default is "image"', default='image', type=str)
    parser.add_argument('-d', '--display', dest='display',
            help='whether display the detection result, default is True', default=True, type=bool)
    parser.add_argument('-g', '--gpu', dest='use_gpu',
            help='True for GPU and False for CPU', default=True, type=bool)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


CLASSES = {1: 'person', 14: 'bench', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon',
           46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot',
           53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 60: 'bed',
           61: 'dining table', 63: 'tv', 64: 'laptop', 68: 'cell phone', 74: 'book'}

list_frames = []

def object_dectection(image, image_name, net):
    # eval
    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)

    # print('Loaded network {:s}'.format(saved_model))
    # timer = Timer()
    # timer.tic()
    scores, boxes = im_detect(net, image)
    # timer.toc()
    # print('Obeject detection took {:.3f}s for {:d} object proposals'.format(
    #     timer.total_time(), boxes.shape[0]))

    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    tmp = []
    rcnn = {}

    for cls_ind, cls in CLASSES.items():
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]

        for det_inst in dets:
            if det_inst[4] > CONF_THRESH:
                inst_tmp = [image_name]
                if cls_ind == 1:
                    inst_tmp.append('Human')
                else:
                    inst_tmp.append('Object')
                inst_tmp.append(det_inst[:4])
                inst_tmp.append(np.nan)
                inst_tmp.append(cls_ind)
                inst_tmp.append(det_inst[4])
                tmp.append(inst_tmp)

    rcnn[image_name] = tmp

    return rcnn

def fetch_frame():
    global list_frames
    ret, frame = cap.read()
    if ret is True:
        if len(list_frames) > 31:
            list_frames.append(frame)
            list_frames.pop(0)
            # print('pop')
        else:
            list_frames.append(frame)
    t = threading.Timer(0.01, fetch_frame)
    t.start()


if __name__ == '__main__':

    args = parse_args()

    if args.type == 'video':
        cap = cv2.VideoCapture(args.demo_file)
    elif args.type == 'camera':
        cap = cv2.VideoCapture(int(args.demo_file))
    else:
        AssertionError('type is not correct')

    prior_mask = pickle.load(open(cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb"), encoding='iso-8859-1')
    Action_dic = json.load(open(cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y: x for x, y in Action_dic.items()}

    # load detection model
    detection_model = 'output/res50_faster_rcnn_iter_1190000.pth'
    if not os.path.isfile(detection_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(detection_model))
    detection_net = resnetv1(num_layers=50)
    detection_net.create_architecture(81, tag='default', anchor_scales=[4, 8, 16, 32], anchor_ratios=[0.5, 1, 2])
    detection_net.load_state_dict(torch.load(detection_model, map_location=lambda storage, loc: storage))

    # load hoi_detection model
    hoi_model = 'output/HOI_iter_250000.pth'
    if not os.path.isfile(hoi_model):
        raise IOError(
            ('{:s} not found.\nDid you download the proper networks from '
             'our server and place them properly?').format(hoi_model))

    # load network
    hoi_net = resnet18()
    if args.use_gpu:
        hoi_net = hoi_net.cuda()
    hoi_net.load_state_dict(torch.load(hoi_model, map_location=lambda storage, loc: storage))

    cap = cv2.VideoCapture(0)

    t = threading.Timer(0.04, fetch_frame)
    t.start()
    time.sleep(1)

    index = -1
    text_list = []
    while (cap.isOpened()):
        index = index + 1
        timer = Timer()
        timer.tic()

        # 4. read image
        if len(list_frames) > 31:
            image = list_frames[30]
        else:
            print("not enough frames")
            time.sleep(0.5)
            continue

        nowtime = datetime.datetime.now().strftime('%H:%M:%S')
        rcnn = object_dectection(image, index, detection_net)
        hoi = hoi_detect(hoi_net, image, index, rcnn, prior_mask, Action_dic_inv, 0.4, 0.9, 3)
        timer.toc()
        print('HOI detection took {:.3f}s'.format(timer.total_time()))

        # Visualization
        vis_detection(image, index, hoi)  # visualization version 1

        # visualization version 2
        for ele in hoi:
            if (ele['image_id'] == index):
                action_count = 0
                H_box = ele['person_box']

            for action_key, action_value in ele.items():
                if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                    if (not np.isnan(action_value[0])) and (action_value[5] >= 0.001):
                        text = nowtime + ':  ' + action_key.split('_')[0] + ' ' + CLASSES[np.int(action_value[4])] + ', ' + "%.2f" % \
                               action_value[5]

                        if len(text_list) > 11:
                            text_list.append(text)
                            text_list.pop(0)
                        else:
                            text_list.append(text)

            cv2.rectangle(image, (int(H_box[0]), int(H_box[1])), (int(H_box[2]), int(H_box[3])), (0, 0, 255), 2)
        for i in range(len(text_list)):
            cv2.putText(image, text_list[i], (0, (i + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)

        cv2.imshow('result', image)
        cv2.waitKey(1)

