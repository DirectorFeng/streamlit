import torch
import torch.nn as nn
from model.unet_model import UNet
from model.Unet_scAG import *
import numpy as np
import cv2


def nodule_predict(input_image, option):
    global net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = UNet(n_channels=1, n_classes=1)
    # net = Effi_UNet(in_channels=1, classes=1)
    # net = UnetPlusPlus(num_classes=1)
    # net = UResnet(block=BottleNeck, layers=[3, 4, 6, 3], num_classes=1)
    # net = AttU_Net()
    # net = R2AttU_Net()
    # net = R2U_Net()
    if option == 'Unet':
        net = torch.load('checkpoints/unet_new.pt')
    elif option == 'AttentionUnet':
        net = torch.load('checkpoints/att_unet_scAG_PT.pt')
    elif option == 'EfficientUnet++':
        net = torch.load('checkpoints/EffiUnetppb7_scse.pt')
    net.to(device=device)
    # net.load_state_dict(torch.load('checkpoints/att_unet_scAG.pth', map_location=device))  # todo
    net.eval()
    input_image_tensor = torch.from_numpy(input_image)
    img_tensor = input_image_tensor.to(device=device, dtype=torch.float32)
    # 预测
    pred = net(img_tensor)
    sigmoid = nn.Sigmoid()
    pred = sigmoid(pred)
    # print(pred)
    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]
    # print(pred)
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)
    return pred