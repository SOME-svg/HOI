# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao, based on code from Zheqi he and Xinlei Chen
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from utils.timer import Timer
from utils.ult import Get_next_sp
from utils.apply_prior import apply_prior


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
import pickle
import numpy as np
import os
import sys
import glob
import time


def hoi_detect(net, im, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag):

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, 3, im_shape[0], im_shape[1])
    im_orig = Variable(torch.from_numpy(im_orig).cuda())

    detection = []
    net.eval()
    H_num = Variable(torch.IntTensor([1]).cuda())

    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human

            # Predict actrion using human appearance only
            H_boxes = np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]).reshape(1, 5)
            H_boxes = Variable(torch.from_numpy(H_boxes).cuda())
            # prediction_H  = net.test_image_H(sess, im_orig, blobs)

            # save image information
            dic = {}
            dic['image_id'] = image_id
            dic['person_box'] = Human_out[2]

            # Predict action using human and object appearance
            Score_obj = np.empty((0, 5 + 8), dtype=np.float32)

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (
                np.all(Object[2] == Human_out[2])):  # This is a valid object
                    # if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])) and (Object[2][3] - Object[2][1]) *  (Object[2][2] - Object[2][0]) > 10000: # This is a valid object

                    O_boxes = np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 5)
                    O_boxes = Variable(torch.from_numpy(O_boxes).cuda())
                    sp = Get_next_sp(Human_out[2], Object[2]).reshape(1, 2, 64, 64).astype(np.float32)
                    sp = Variable(torch.from_numpy(sp).cuda())
                    cls_score_H, cls_score_O, cls_score_Hsp = net(im_orig, sp, H_boxes, O_boxes, H_num)
                    prediction_HO = cls_score_Hsp * (cls_score_H + cls_score_O)
                    # prediction_HO = F.softmax(prediction_HO,dim=1)
                    prediction_HO = prediction_HO.cpu().detach().numpy()
                    if prior_flag == 1:
                        prediction_HO = apply_prior(Object, prediction_HO)
                    if prior_flag == 2:
                        prediction_HO = prediction_HO * prior_mask[:, Object[4]].reshape(1, 8)
                    if prior_flag == 3:
                        prediction_HO = apply_prior(Object, prediction_HO)
                        prediction_HO = prediction_HO * prior_mask[:, Object[4]].reshape(1, 8)

                    This_Score_obj = np.concatenate(
                        (Object[2].reshape(1, 4), np.array(Object[4]).reshape(1, 1), prediction_HO * np.max(Object[5])),
                        axis=1)
                    Score_obj = np.concatenate((Score_obj, This_Score_obj), axis=0)

            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if Score_obj.shape[0] == 0:
                continue

            # Find out the object box associated with highest action score
            max_idx = np.argmax(Score_obj, 0)[5:]

            for i in range(8):
                if np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i] == 0:
                    dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1, 4),
                                                       np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i])

                # Action with >0 score
                else:
                    dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:5],
                                                       np.max(Human_out[5]) * Score_obj[max_idx[i]][5 + i])

            detection.append(dic)

    return detection


# def test_net(net, Test_RCNN, prior_mask, Action_dic_inv, img_dir, output_dir, object_thres, human_thres, prior_flag):
#
#
#     np.random.seed(cfg.RNG_SEED)
#     detection = []
#     count = 0
#
#     # timers
#     _t = {'im_detect' : Timer(), 'misc' : Timer()}
#
#
#     for im_file in glob.glob(img_dir + "*.png"):
#
#
#         _t['im_detect'].tic()
#
#         image_id   = im_file.split('\\')[-1]
#
#         im_detect(net, img_dir, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection)
#
#         _t['im_detect'].toc()
#
#         print('im_detect: {:d} {:.3f}s'.format(count + 1, _t['im_detect'].average_time))
#         count += 1
#
#     pickle.dump( detection, open( output_dir, "wb"))