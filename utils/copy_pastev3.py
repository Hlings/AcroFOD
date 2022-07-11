import random
import cv2
import numpy as np
import torch
import math

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

def copy_paste(im, labels, cp_type, p=0.9): 
    """
    labels     0      xmin        ymin        xmax        ymax
    [
    [          0      355.82      122.24      407.64      259.75]
                                                                ]  
    img[ymin:ymax, xmin:xmax]   w = int(xmax-xmin)  h = int(ymax-ymin) 
    """
    n = len(labels)
    if n <= 1:
        return im,labels

    #change_times = int(n*p)
    cls_unique = np.unique(labels[:,0])
    #print("n is", n)
    for cls_number in cls_unique:
        cls_bool_array = np.where(labels[:,0]==cls_number)
        labels_refine = labels[cls_bool_array]        
        m = len(labels_refine)
        
        if m <= 1: # if number of labels <= 1 then no any exchange
            continue
        for times in range(int(m*p)+6):   
            cp_labels = random.sample(labels_refine.tolist(),2) 
            im = change_target(im, cp_labels, labels_refine, cp_type) 
    
    return im, labels


def change_target(im, cp_labels, labels, cp_type, ioa_th=0.3, margin=0.5): 
    #_, w, _ = im.shape
    box1 = np.array([cp_labels[0][1], cp_labels[0][2], cp_labels[0][3], cp_labels[0][4]])
    box2 = np.array([cp_labels[1][1], cp_labels[1][2], cp_labels[1][3], cp_labels[1][4]])
    
    _, x1a, y1a, x1b, y1b = np.array(cp_labels[0], dtype=int)
    _, x2a, y2a, x2b, y2b = np.array(cp_labels[1], dtype=int)   
    
    w1, h1, w2, h2 = int(x1b-x1a), int(y1b-y1a), int(x2b-x2a), int(y2b-y2a)
    alpha = (w1/(h1+1e-6)) / ((w2/(h2+1e-6))+1e-6) 
    if alpha < 1 - margin or alpha > 1 + margin :
        return im
    
    
    labels_moved_1, labels_moved_2 = labels.tolist().copy(), labels.tolist().copy() 
    labels_moved_1.remove(cp_labels[0])
    labels_moved_2.remove(cp_labels[1])
    
    gt_1 = np.array(labels_moved_1)[:, 1:5]
    gt_2 = np.array(labels_moved_2)[:, 1:5]
    
    ioa1 = bbox_ioa(box1, gt_1)
    ioa2 = bbox_ioa(box2, gt_2)
   
    
    if (ioa1 < ioa_th).all() and (ioa2 < ioa_th).all():      
        target1 = im[y1a:y1b, x1a:x1b]
        target2 = im[y2a:y2b, x2a:x2b]
    
        if (x2b-x2a) <= 0 or (y2b-y2a) <= 0 or (x1b-x1a) <= 0 or (y1b-y1a) <= 0:
            return im
    
        target1 = cv2.resize(target1, (x2b-x2a, y2b-y2a), interpolation=cv2.INTER_LINEAR)
        target2 = cv2.resize(target2, (x1b-x1a, y1b-y1a), interpolation=cv2.INTER_LINEAR)
    
        if cp_type > 0 and cp_type < 1: # simple copy-paste
            im[y1a:y1b, x1a:x1b] = target2    #+ 0.5 * im1[y1a:y1b, x1a:x1b]
            im[y2a:y2b, x2a:x2b] = target1    #+ 0.5 * im2[y2a:y2b, x2a:x2b]
            return im
        
        if cp_type > 1 and cp_type < 2: #gaussian copy-paste
            h_t1, w_t1 = int(im.shape[0]/4), int(im.shape[1]/4)
            g_map_t1 = _gaussian_map(h_t1, w_t1, x1a, y1a, x1b, y1b)
            g_map_t2 = _gaussian_map(h_t1, w_t1, x2a, y2a, x2b, y2b)
            im[y1a:y1b, x1a:x1b] = target2 * g_map_t1 + im[y1a:y1b, x1a:x1b] * (1 - g_map_t1)
            im[y2a:y2b, x2a:x2b] = target1 * g_map_t2 + im[y2a:y2b, x2a:x2b] * (1 - g_map_t2)
            return im
        
        if cp_type > 2 and cp_type < 3.0: # box-level copy-paste
            r = np.random.beta(8.0, 8.0)
            im[y1a:y1b, x1a:x1b] = target2 * r + (im[y1a:y1b, x1a:x1b] * (1-r))
            im[y2a:y2b, x2a:x2b] = target1 * r + (im[y2a:y2b, x2a:x2b] * (1-r))
            return im
    return im

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