from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck',
           'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter','bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite','baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
           'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier','toothbrush')
FONT = cv2.FONT_HERSHEY_SIMPLEX
color = [ (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 255, 255),
         (160, 82, 45), (128, 42, 42), (255, 128, 0)]


def vis_detection(im_data, image_id, detection):
    # print('showing result')


    HO_dic = {}
    HO_set = set()
    count = -1
    is_sitting = False
    # im_data = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)
    # im_data = np.array(im_data)
    # im_data = Image.fromarray(im_data)

    for ele in detection:
        if (ele['image_id'] == image_id):
            action_count = 0
            H_box = ele['person_box']

            if tuple(H_box) not in HO_set:
                HO_dic[tuple(H_box)] = count
                HO_set.add(tuple(H_box))
                count += 1

            # show_H_flag = 1
            for action_key, action_value in ele.items():
                if (action_key.split('_')[-1] != 'agent') and action_key != 'image_id' and action_key != 'person_box':
                    if (not np.isnan(action_value[0])) and (action_value[5] >= 0.001):

                        O_box = action_value[:4]

                        action_count += 1

                        if tuple(O_box) not in HO_set:
                            HO_dic[tuple(O_box)] = count
                            HO_set.add(tuple(O_box))
                            count += 1


                        # cv2.rectangle(im_data, int(H_box[0]), int(H_box[1]), int(H_box[2]), int(H_box[3]), cc()[:3]*255, 2)
                        text = action_key.split('_')[0] + ' ' + CLASSES[np.int(action_value[4])] + ', ' + "%.2f" % action_value[5]
                        # cv2.putText(im_data, text, (int(H_box[0]), int(H_box[1] + (action_count+1) * 30)), FONT, 0.5, color[HO_dic[tuple(H_box)]], 2)
                        cv2.putText(im_data, text, (0, (action_count + 1) * 30), FONT, 0.5, color[HO_dic[tuple(O_box)]], 2)
                        # cv2.rectangle(im_data, (int(O_box[0]), int(O_box[1])), (int(O_box[2]), int(O_box[3])), color[HO_dic[tuple(O_box)]], 1)
            cv2.rectangle(im_data, (int(H_box[0]), int(H_box[1])), (int(H_box[2]), int(H_box[3])), color[HO_dic[tuple(H_box)]], 1)

    cv2.imshow('result', im_data)
    cv2.waitKey(1)
    # cv2.imwrite('demo/'+'pic'+str(image_id)+'.png', im_data)



