# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
import  os
# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
# net =models.resnet18(pretrained=True)
# for idx ,m in net.named_modules():
#     print(idx)
# print(net)
#classifier
#features.36
#avgpool







directory_name = 'E:/imagenet_val'
save_dir_name = 'E:/imagenet_val_cam'
for filename in os.listdir(directory_name):
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
    net.eval()
    # hook the feature extractor
    features_blobs = []


    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())


    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        # size_upsample = (256, 256)
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    # img = cv2.imread(directory_name + "/" + filename)
    image_path = directory_name + "/" + filename
    image_savepath = save_dir_name + "/" + filename
    img_pil = Image.open(image_path)
    img_pil=img_pil.convert("RGB")
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    # cv2.imwrite(image_savepath, cv2.resize(np.uint8(255 * cam), (width, height)))
    # #
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(image_savepath, result)
