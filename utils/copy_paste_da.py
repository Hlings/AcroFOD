import random
import cv2
import numpy as np
import torch
import math


def copy_paste_api(imgs_source, imgs_target, labels_source,
                   labels_target):  

    for number_1 in range(0, len(imgs_source)):
        number_2 = random.randint(0, int(max(labels_target[:, 0])))
        img1 = imgs_source[number_1].permute(1, 2, 0)   # (C, 640, 360) to (640, 360, C)
        img2 = imgs_target[number_2].permute(1, 2, 0)
        w, h = img1.shape[0], img1.shape[1]
        labels1 = xywh2xyxy(labels_source[labels_source[:, 0] == number_1, 1:6], w, h)
        labels2 = xywh2xyxy(labels_target[labels_target[:, 0] == number_2, 1:6], w, h)

        img1, img2, labels1_xyxy, labels2_xyxy = copy_paste(np.array(img1), np.array(img2), labels1, labels2)

        imgs_source[number_1] = torch.tensor(img1).permute(2, 0, 1)  # (640, 360, C) to (C, 640, 360)
        imgs_target[number_2] = torch.tensor(img2).permute(2, 0, 1)

    return imgs_source, imgs_target, labels_source, labels_target


def copy_paste(im1, im2, labels1, labels2, p=0.6):  # img im shape [H, W, C]
    """
    labels     0      xmin        ymin        xmax        ymax
    [
    [          0      355.82      122.24      407.64      259.75]
                                                                ]
    img[ymin:ymax, xmin:xmax]   w = int(xmax-xmin)  h = int(ymax-ymin)
    """
    n1, n2 = len(labels1), len(labels2)
    if n1 < 1 or n2 < 1:
        return im1, im2, labels1, labels2

    # change_times = int(n*p)
    cls_unique = np.unique(labels1[:, 0])
    # print("n is", n)
    for cls_number in cls_unique: 

        labels1_refine = labels1[np.where(labels1[:, 0] == cls_number)]
        labels2_refine = labels2[np.where(labels2[:, 0] == cls_number)]
        
        m1, m2 = len(labels1_refine), len(labels2_refine)

        if m1 < 1 or m2 < 1:  # if number of labels <= 1 then no any exchange
            continue
        for times in range(int(m1 * p) + 3):
            cp_labels = random.sample(labels1_refine.tolist(), 1) + random.sample(labels2_refine.tolist(), 1)                                   
            #im1, im2 = change_target(im1, im2, cp_labels, labels1_refine, labels2_refine)  
            im1, im2 = change_target(im1, im2, cp_labels, labels1_refine, labels2_refine)  
            
    return im1, im2, labels1, labels2

"""
def change_target(im1, im2, cp_labels, gaussian=False):  # 

    _, x1a, y1a, x1b, y1b = np.array(cp_labels[0], dtype=int)
    _, x2a, y2a, x2b, y2b = np.array(cp_labels[1], dtype=int)

    target1 = im1[y1a:y1b, x1a:x1b]
    target2 = im2[y2a:y2b, x2a:x2b]

    if (x2b - x2a) <= 0 or (y2b - y2a) <= 0 or (x1b - x1a) <= 0 or (y1b - y1a) <= 0:
        return im1, im2
    
    target1 = cv2.resize(target1, (x2b - x2a, y2b - y2a), interpolation=cv2.INTER_LINEAR)
    target2 = cv2.resize(target2, (x1b - x1a, y1b - y1a), interpolation=cv2.INTER_LINEAR)
    if not gaussian:
        im1[y1a:y1b, x1a:x1b] = target2    #+ 0.5 * im1[y1a:y1b, x1a:x1b]
        im2[y2a:y2b, x2a:x2b] = target1    #+ 0.5 * im2[y2a:y2b, x2a:x2b]
    else:
        h_t1, w_t1, h_t2, w_t2 = im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]
        g_map_t1 = _gaussian_map(h_t1, w_t1, x1a, y1a, x1b, y1b)
        g_map_t2 = _gaussian_map(h_t2, w_t2, x2a, y2a, x2b, y2b)
        im1[y1a:y1b, x1a:x1b] = target2 * g_map_t1 + im1[y1a:y1b, x1a:x1b] * (1 - g_map_t1)
        im2[y2a:y2b, x2a:x2b] = target1 * g_map_t2 + im2[y2a:y2b, x2a:x2b] * (1 - g_map_t2)

    return im1, im2
"""

def change_target(im1, im2, cp_labels, labels1_refine, labels2_refine, ioa_th=0.4, margin=0.6,
                  gaussian=False): 
    # _, w, _ = im.shape

    box1 = np.array([cp_labels[0][1], cp_labels[0][2], cp_labels[0][3], cp_labels[0][4]])
    box2 = np.array([cp_labels[1][1], cp_labels[1][2], cp_labels[1][3], cp_labels[1][4]])

    _, x1a, y1a, x1b, y1b = np.array(cp_labels[0], dtype=int)
    _, x2a, y2a, x2b, y2b = np.array(cp_labels[1], dtype=int)

    w1, h1, w2, h2 = int(x1b - x1a), int(y1b - y1a), int(x2b - x2a), int(y2b - y2a)
    alpha = (w1 / (h1 + 1e-6)) / ((w2 / (h2 + 1e-6)) + 1e-6)
    if alpha < 1 - margin or alpha > 1 + margin:
        return im1, im2
    
    ioa1 = get_bbox_ioa(box1, labels1_refine, cp_labels[0])
    ioa2 = get_bbox_ioa(box2, labels2_refine, cp_labels[1])
    
    if (ioa1 < ioa_th).all() and (ioa2 < ioa_th).all():
        target1 = im1[y1a:y1b, x1a:x1b]
        target2 = im2[y2a:y2b, x2a:x2b]

        if (x2b - x2a) <= 0 or (y2b - y2a) <= 0 or (x1b - x1a) <= 0 or (y1b - y1a) <= 0:
            return im1, im2

        target1 = cv2.resize(target1, (x2b - x2a, y2b - y2a), interpolation=cv2.INTER_LINEAR)
        target2 = cv2.resize(target2, (x1b - x1a, y1b - y1a), interpolation=cv2.INTER_LINEAR)

        if not gaussian:
            im1[y1a:y1b, x1a:x1b] = target2  # + 0.5 * im1[y1a:y1b, x1a:x1b]
            im2[y2a:y2b, x2a:x2b] = target1  # + 0.5 * im2[y2a:y2b, x2a:x2b]
        else:
            h_t1, w_t1, h_t2, w_t2 = im1.shape[0], im1.shape[1], im2.shape[0], im2.shape[1]
            g_map_t1 = _gaussian_map(h_t1, w_t1, x1a, y1a, x1b, y1b)
            g_map_t2 = _gaussian_map(h_t1, w_t1, x2a, y2a, x2b, y2b)
            im1[y1a:y1b, x1a:x1b] = target2 * g_map_t1 + im1[y1a:y1b, x1a:x1b] * (1 - g_map_t1)
            im2[y2a:y2b, x2a:x2b] = target1 * g_map_t2 + im2[y2a:y2b, x2a:x2b] * (1 - g_map_t2)

    return im1, im2


def get_bbox_ioa(box, labels_refine, label_to_move):
    gt = labels_refine.tolist().copy()
    
    if len(gt) == 1:
        gt = np.array([0.0, 0.0, 1.0, 1.0])
        ioa = bbox_ioa(box, gt)
    else:
        gt.remove(label_to_move)
        gt = np.array(gt)[:, 1 : 5]
        ioa = bbox_ioa(box, gt)
    
    return ioa
        

def _gaussian_map(height, width, x1, y1, x2, y2):
    """
    Args:
        height: h of total img
        width: w of total img
        x1 y1 x2 y2 : bbx coord

    Returns: 2-D gaussian_map for bbx (numpy array)
    """
    h, w = int(y2 - y1), int(x2 - x1)

    mean_torch = torch.tensor([h // 2, w // 2])  

    # r_var = (height * width / (2 * math.pi)) ** 0.5  # * 0.8  
    r_var = (h * w * 4 / (2 * math.pi)) ** 0.5
    var_x = torch.tensor([(h / height) * r_var], dtype=torch.float32)
    var_y = torch.tensor([(w / width) * r_var], dtype=torch.float32)

    x_range = torch.arange(0, h, 1)
    y_range = torch.arange(0, w, 1)
    xx, yy = torch.meshgrid(x_range, y_range)
    pos = torch.empty(xx.shape + (2,))
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    g_map = torch.exp(
        torch.tensor(-(((xx.float() - mean_torch[0]) ** 2 / (2.0 * var_x ** 2) + (yy.float() - mean_torch[1])
                        ** 2 / (2.0 * var_y ** 2)))))  # shape(h,w)

    return np.expand_dims(np.array(g_map), axis=2)


def xywh2xyxy(x, w, h):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * w  # top left x
    y[:, 2] = (x[:, 2] - x[:, 4] / 2) * h  # top left y
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * w  # bottom right x
    y[:, 4] = (x[:, 2] + x[:, 4] / 2) * h  # bottom right y
    return y


def xyxy2xywh(x, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 1] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 2] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 3] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 4] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area