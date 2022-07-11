import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device, intersect_dicts
from models.yolo_feature import Model_feature

def get_feature(img, model): 
    model.eval()
    img_feature = model(img)[1] # torch.tensor [B 1280 H/32 W/32]
    img_feature = img_feature.mean(3).mean(2)
    return img_feature  # torch.tensor [B, 1280] feature

def MMD_distance(f_S, f_T, k): # S: [B1 1280] T: [B2 1280] 
    f_T = f_T - f_S.mean(0)
    f_T = f_T.mul(f_T) 

    f_T = f_T.sum(dim=1)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] 

def cosine_distance(f_S, f_T, k): # f_S: [B1 1280]  T: [B2 1280]
    dist_tensor = []
    for i in f_T:
        i = i.unsqueeze(0)
        dist = (1-torch.cosine_similarity(f_S,i,dim=1)).sum()
        dist_tensor.append(dist)
        
    f_T = torch.tensor(dist_tensor)

    if f_T.shape[0] > k:
        T_topk = f_T.topk(k=k, largest = False)
    else: 
        return torch.arange(0, f_T.shape[0])
    return T_topk[1] # T中和 

def choice_topk(imgs, targets, paths, topk_index):
    paths_refine = []
    labels_refine = torch.tensor([])

    for i in list(topk_index):
        paths_refine.append(paths[i])
        
    for index in topk_index:
        t = targets[targets[:, 0] == index, :]
        if t.shape[0] > 0:
            number = torch.nonzero((topk_index == t[0][0]))[0][0]
            t[:, 0] = number
        labels_refine = torch.cat((labels_refine, t), dim = 0)
    
    imgs_refine = imgs[topk_index, :, :, :]
    return imgs_refine, labels_refine, paths_refine 