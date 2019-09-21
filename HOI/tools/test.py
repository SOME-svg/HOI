from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import cv2
import time
import threading
import datetime

list_frames = []


def fetch_frame():
    ret, frame = cap.read()
    if ret is True:
        if len(list_frames) > 31:
            list_frames.append(frame)
            list_frames.pop(0)
        else:
            list_frames.append(frame)
    t = threading.Timer(0.03, fetch_frame)
    t.start()


def delay():
    i = 0
    while(i < 10000000):
        i = i + 1

    return 'delay'

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    t = threading.Timer(0.03, fetch_frame)
    t.start()

    text = ''
    while (cap.isOpened()):
        if len(list_frames) > 31:
            image = list_frames[30]
        else:
            print("not enough frames")
            time.sleep(0.5)
            continue
        # print('test')
        # for i in range(5):
        #     delay()
        nowtime = datetime.datetime.now().strftime('%H:%M:%S')
        text = nowtime
        # text = nowtime
        cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.imshow('video', image)
        cv2.waitKey(1)

