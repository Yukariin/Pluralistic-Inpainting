import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from random import randint
import numpy as np
import cv2
from PIL import Image
import random

###################################################################
# random mask generation
###################################################################


def random_regular_mask(img):
    """Generates a random regular hole"""
    mask = torch.ones_like(img)
    s = img.size()
    N_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (N_mask + 1)
    limy = s[2] - s[2] / (N_mask + 1)
    for _ in range(N_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (N_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (N_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img):
    """Generates a center hole with 1/4*W and 1/4*H"""
    mask = torch.ones_like(img)
    size = img.size()
    x = int(size[1] / 4)
    y = int(size[2] / 4)
    range_x = int(size[1] * 3 / 4)
    range_y = int(size[2] * 3 / 4)
    mask[:, x:range_x, y:range_y] = 0

    return mask


def random_irregular_mask(img,
                          min_stroke=1, max_stroke=4,
                    min_vertex=1, max_vertex=12,
                    min_length_divisor=10, max_length_divisor=18,
                    min_brush_width_divisor=16, max_brush_width_divisor=10):
    """Generates a random irregular mask with lines, circles and elipses"""
    transform = transforms.Compose([transforms.ToTensor()])
    mask = torch.ones_like(img)
    size = img.size()
    img = np.zeros((size[1], size[2], 1), np.uint8)

    # Set size scale
    max_width = 20
    if size[1] < 64 or size[2] < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    min_length = size[1] // min_length_divisor
    max_length = size[1] // max_length_divisor
    min_brush_width = size[1] // min_brush_width_divisor
    max_brush_width = size[1] // max_brush_width_divisor
    max_angle = 2*np.pi
    num_stroke = np.random.randint(min_stroke, max_stroke+1)

    for _ in range(num_stroke):
        num_vertex = np.random.randint(min_vertex, max_vertex+1)
        start_x = np.random.randint(size[1])
        start_y = np.random.randint(size[2])

        for _ in range(num_vertex):
            angle = np.random.uniform(max_angle)
            length = np.random.uniform(min_length, max_length)
            brush_width = np.random.randint(min_brush_width, max_brush_width+1)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(img, (start_y, start_x), (end_y, end_x), (1, 1, 1), brush_width)

            start_x, start_y = end_x, end_y

    img = img.reshape(size[2], size[1])
    img = Image.fromarray(img*255)

    img_mask = transform(img)
    for j in range(size[0]):
        mask[j, :, :] = img_mask < 1

    return mask

###################################################################
# multi scale for image generation
###################################################################


def scale_img(img, size):
    scaled_img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
    return scaled_img


def scale_pyramid(img, num_scales):
    scaled_imgs = [img]

    s = img.size()

    h = s[2]
    w = s[3]

    for i in range(1, num_scales):
        ratio = 2**i
        nh = h // ratio
        nw = w // ratio
        scaled_img = scale_img(img, size=[nh, nw])
        scaled_imgs.append(scaled_img)

    scaled_imgs.reverse()
    return scaled_imgs
