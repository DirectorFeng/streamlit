import cv2
import math
import pandas as pd
import streamlit as st
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from PIL import Image


colors = [(0, 0, 255), '红',
          (0, 255, 0), '绿',
          (0, 255, 255), '黄',
          (255, 255, 0), '青',
          (255, 0, 255), '紫',
          (255, 0, 0), '蓝']
params = []


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef2(output, target):
    "This metric is for validation purpose"
    smooth = 1e-5

    output = output.view(-1)
    output = (output > 0.5).float().cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)

# @st.cache_data
def contours_norm_compute(img):
    img_ = img.copy()
    ret, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite("./result/dst.png", dst)
    contours, hierachy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_ = cv2.cvtColor(img_, cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(img, contours[i], -1, colors[i*2 % len(colors)], 2)
        area = round(cv2.contourArea(contours[i]), 1)
        length = round(cv2.arcLength(contours[i], True), 1)
        print(f'{colors[(i*2+1) % len(colors)]}色框区域结节的面积为：{area}')
        print(f'{colors[(i*2+1) % len(colors)]}色框区域结节的周长为：{length}')
        r = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(r)
        box = np.int0(box)
        cv2.drawContours(img_, [box], 0, colors[i*2 % len(colors)], 2)
        M = cv2.moments(contours[i])
        # print(m)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'{colors[(i*2+1) % len(colors)]}色框区域结节的质心为：[{cx}, {cy}]')
        diam = round(math.sqrt(4 * area / math.pi), 1)
        print(f'{colors[(i*2+1) % len(colors)]}色框区域结节的最大直径为：{diam}')
        params.append([colors[(i*2+1) % len(colors)], (cx, cy), area, diam])
    df = pd.DataFrame(columns=["结节区域", "中心坐标/(x,y)", "结节大小/mm²", "最大直径/mm"])
    for i in range(len(params)):
        new_row = [params[i][0], params[i][1],
                   params[i][2], params[i][3]]
        df.loc[len(df)] = new_row
    cv2.imwrite("./result/contour.png", img)
    cv2.imwrite("./result/minAreaRect.png", img_)
    img_[dst == 255] = 0
    cv2.imwrite("./result/rect.png", img_)
    return params
