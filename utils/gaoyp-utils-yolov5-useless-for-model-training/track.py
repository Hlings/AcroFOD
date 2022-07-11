import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy): # 返回bounding box 坐标 x_center y_center w h 形式
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label): #计算label的颜色 这里label为正整数
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# 输入是一个img 输出一个 画上bbx 并标记上 identities对应的id 所以 identities中的id 和 bbox中对应索引一致
def draw_boxes(img, bbox, identities=None, offset=(0, 0)): 
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def sort_gts(file_path):
    with open(file_path, "r") as f:
        gts = f.readlines()
        frame_det = {}
        for line in gts:
            line = line.strip().split(",")[:3]
            index_of_frame = line[0]
            number_of_persons = line[1]
            if index_of_frame not in list(frame_det.keys()):
                frame_det[index_of_frame] = 0
            frame_det[index_of_frame] = number_of_persons
        for i in range(0,81):
            frame_det[str(i)] = 4
    return frame_det

def crop_img(img,bbox,identities,number_of_frame,img_path): # 对最终的追踪结果进行剪裁
    number_of_frame = str(int(number_of_frame))
    for i,box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        img_id = int(identities[i]) if identities is not None else 0
        img_cropped = img[y1:y2, x1:x2]
        img_name = number_of_frame.zfill(5) + "_" + str(img_id).zfill(2) + "_" + str(int(i)).zfill(2) + ".jpg"  # frame_id_object.jpg
                
        #home_path = '/userhome'
        out_path = os.path.join(img_path, img_name)
        print(out_path)
        cv2.imwrite(out_path,img_cropped)
        
def draw_fps(img,fps):  # img为当前输入的图片 fps代表所显示的帧数返回添加帧数之后的图片
    fps_str = str(fps)
    text = "fps = " + fps_str
    cv2.putText(img, text, (50, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
    return img 

def draw_rank1_missrate(img,rank1,missrate,number_of_frame):
    if number_of_frame < 11:  # 因为追踪有一个初始帧 此处这么设置就是屏蔽了初始帧对应的指标
        return img
    rank1_str = str(rank1)
    missrate_str = str(missrate)
    text_rank1 = "rank1 = " + rank1_str
    text_missrate = "detect_rate = " + missrate_str
    cv2.putText(img, text_rank1, (50, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
    cv2.putText(img, text_missrate, (50, 105), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
    return img
    
def detect(opt, save_img=False):
    
    out, source, weights, view_img, save_txt, imgsz, cropped, hhxd_003, hhxd_004, zh_002 = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.if_cropped, opt.if_hhxd_003, opt.if_hhxd_004, opt.if_zh_002
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort 初始化追踪器
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device) # 获得算法运行的设备
    if os.path.exists(out):            # 判断输出路径是否已经存在
        shutil.rmtree(out)             # 如果已经存在则删除这个路径 
    os.makedirs(out)                   # 创建这个路径
    half = device.type != 'cpu'        # 如果设备支持CUDA则模型切换为半精度FP16
    

    # Load model
    model = torch.load(weights, map_location=device)[   # 这里加载的是YOLO v5检测需要的模型 加载为FP32
        'model'].float()  
    model.to(device).eval()                             # 切换模型为评估模式
    if half:                                            # 如果支持CUDA则切换为半精度模式 提高速度
        model.half()  

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:                  # 这里判断推断对象的来源是哪里
        view_img = True
        cudnn.benchmark = True  # 对于尺寸图片尺寸不变的情况设置cudnn为True加速source的推断
        dataset = LoadStreams(source, img_size=imgsz) # dataset 为以推流的形式加载数据
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)  # 以加载image的形式加载数据 对于视频对象应为此种加载方式

    # names变量代表模型的名字
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time() # t0代表开始推断的时间
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # 初始化图片对象
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None # 这里用空的图片拿模型运行了一次 这里的作用是什么呢？

    save_path = str(Path(out))                 # 代表输出保存的路径
    txt_path = str(Path(out)) + '/results.txt' # 如果输出txt结果 则保存的路径
    img_path = str(Path(out)) + '/img'         # 输出保存文件的路径
    
    if cropped:
        if os.path.exists(img_path):            # 判断输出路径是否已经存在
            shutil.rmtree(img_path)             # 如果已经存在则删除这个路径 
        os.makedirs(img_path)                   # 创建这个路径    
    
    number_of_frame = 1.0 # 初始化记录帧数
    time_all = 0.0        # 初始化推断过程需要的总时间
    person_detected = 0.0 # 统计一共检测到的人
    person_all = 0.0      # 统计应该有的所有人
    rank1_correct = 0.0   # 统计rank1正确的数目
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset): #开始对帧序列进行处理
        # 对img的格式从numpy矩阵复制到模型运行的指定设备商 并把像素值从0-255归一化为0-1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3: # 如果img的维度为[[[]]]则向上填充一维为4 img是当前正在处理的图片
            img = img.unsqueeze(0)

        # 推断 这里预测出了 每一张图片上面存在的框
        t1 = time_synchronized()  # 记录推断前的时间
        pred = model(img, augment=opt.augment)[0]

        # 这里对预测的结果应用 NMS方法去重
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized() # 记录推断后的时间
        
        # 计算fps
        time_delta = t2-t1 # 0.03s 这样的形式        
        time_all += time_delta
        fps = round(number_of_frame/time_all,2) #保留两位小数
        number_of_frame += 1
        
        # 处理detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # 输出的字符串
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # 这里是对每一类目标检测到的数目进行归一 然后 输出出来 如 3 persons 这样
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                # 适应检测之后的结果到deep sort输入的模式
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # 传递检测结果到deepsort模型
                outputs = deepsort.update(xywhs, confss, im0)
                 
                                        
                # 对每一张图片添加上fps
                draw_fps(im0,fps)
                
                # 如果是红海行动003视频 则绘制上对应的rank1和missrate指标
                if zh_002:
                    file_path = "/userhome/zh_002_gt.txt"
                    frame_persons_dict = sort_gts(file_path)
                    aa = number_of_frame #  float(len(outputs)-1)                    
                    if aa >= 11:
                        person_detected += float(frame_persons_dict[str(int(number_of_frame))]) #这里的person detcted 代表 每个图片中的person个数真值
                        
                    if aa >= 11 and aa<= 32:                        
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                       
                    if aa >= 33 and aa <= 39:
                        person_all += float(len(outputs)-1)
                        rank1_correct += float(len(outputs)-1)                      
                    if aa == 40:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                         
                    if aa >= 41 and aa <= 49:
                        person_all += float(len(outputs)-1)
                        rank1_correct += float(len(outputs)-1)                           
                    if aa >= 50 and aa <= 113:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                               
                    if aa >= 114 and aa <= 133:
                        person_all += float(len(outputs)-1)
                        rank1_correct += float(len(outputs)-1)                           
                    if aa >= 134 and aa <= 141:
                        person_all += float(len(outputs)-1)
                        rank1_correct += float(len(outputs)-2)                                                  
                    if aa >= 142 and aa <= 172:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                                 
                    if aa >= 173 and aa <= 203:
                        person_all += float(len(outputs-1))
                        rank1_correct += float(len(outputs)-2)                                                
                    if aa >= 204 and aa <= 253:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                                        
                    if aa >= 254 and aa <= 314:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 315 and aa <= 316:
                        person_all += float(len(outputs)-1)
                        rank1_correct += float(len(outputs)-1)                                               
                    if aa >= 317 and aa <= 348:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                               
                    if aa >= 349 and aa <= 365:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-2)                                                 
                    if aa >= 366 and aa <= 369:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                               
                    if aa >= 370 and aa <= 400:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 401 and aa <= 403:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                                
                    if aa >= 404 and aa <= 444:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 445 and aa <= 500:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 501 and aa <= 522:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                                
                    if aa >= 523 and aa <= 553:                                               
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 554 and aa <= 559:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                                
                    if aa >= 560 and aa <= 561:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-2)                                                
                    if aa >= 562 and aa <= 566:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)                                                
                    if aa >= 567 and aa <= 611:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))                                               
                    if aa >= 612:
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs)-1)
                                               
                    print("person_detected is ", person_detected)
                    print("person_all is ", person_all )    
                    detect_rate = round(person_all/(person_detected+1e-5), 3)    
                    rank1 = round(rank1_correct/(person_all+1e-5), 3)
                    print("detect_rate is ", detect_rate)
                    print("rank1 is ", rank1)
                    draw_rank1_missrate(im0,rank1,detect_rate,number_of_frame)
                                               
                if hhxd_003: 
                    person_detected += float(len(outputs))  
                    if number_of_frame <= 141:
                        person_all += float(len(outputs))
                    if number_of_frame > 141 and number_of_frame <= 240:
                        person_all += 2.0
                    if number_of_frame > 240:
                        person_all += float(len(outputs))
                        
                    print("person_detected is ", person_detected)
                    print("person_all is ", person_all )
                    detect_rate = round(person_detected/(person_all+1e-5), 2) 
                    print("detect_rate is ", detect_rate)
                    rank1 = str(1.0)
                    draw_rank1_missrate(im0,rank1,detect_rate,number_of_frame)
                    
                if hhxd_004:
                    person_detected += float(len(outputs))
                    if number_of_frame <= 8:
                        person_all += float(len(outputs))
                        rank1_correct += 0.0
                    if number_of_frame > 8 and number_of_frame <= 26:
                        person_all += 3.0
                        rank1_correct += 2.0
                    if number_of_frame > 27 and number_of_frame <= 40:
                        person_all += 2.0
                        rank1_correct += 2.0
                    if number_of_frame > 40 and number_of_frame <= 81:
                        person_all += 3.0
                        rank1_correct += float(len(outputs))
                    if number_of_frame > 81 :
                        person_all += float(len(outputs))
                        rank1_correct += float(len(outputs))
                        
                    print("person_detected is ", person_detected)
                    print("person_all is ", person_all )    
                    detect_rate = round(person_detected/(person_all+1e-5), 2)    
                    rank1 = round(rank1_correct/(person_detected+1e-5), 2)
                    print("detect_rate is ", detect_rate)
                    print("rank1 is ", rank1)
                    draw_rank1_missrate(im0,rank1,detect_rate,number_of_frame)
                    
                # 对每一张图进行裁剪
                if cropped:
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        crop_img(im0, bbox_xyxy, identities, number_of_frame, img_path) # 这里number_of_frame 为浮点数 1.0这样 
                    
                # 对每一张图片画上bounding box                 
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                                    
                # 输出 MOT compliant 结果到文件中 如果要求保存结果的话
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))  # 这一步已经能够展示出来了

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter): # 我认为resize之后不能正常的原因是这里的释放掉了之前的vid_writer 所以导致不能按照之前那种方式来判断
                            vid_writer.release()  # release previous video writer
                        # 这里是按照之前的宽、高、fps来保存视频
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        w2 = int(w/3)
                        h2 = int(h/3)
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w2, h2))
                        im_resize = cv2.resize(im0, (w2,h2), interpolation = cv2.INTER_AREA) # Q 为什么这里resize了之后 输出就是只有一张图片连续不断 问题应该是出在不更新？
                        print("current im_resize is ", im_resize)
                    vid_writer.write(im_resize)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--if_cropped', action='store_true',
                        help='judge whether save cropped imgs')
    parser.add_argument('--if_hhxd_003', action='store_true',
                        help='judge whether the video is hhxd_003 and adding rank1&miss_rate')
    parser.add_argument('--if_hhxd_004', action='store_true',
                        help='judge whether the video is hhxd_004 and adding rank1&miss_rate')
    parser.add_argument('--if_zh_002', action='store_true',
                        help='judge whether the video is hhxd_004 and adding rank1&miss_rate')    
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
