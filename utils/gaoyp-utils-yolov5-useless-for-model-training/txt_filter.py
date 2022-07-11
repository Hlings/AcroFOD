import os
import shutil


def txt_filter(in_path,out_path):
    
    txts = os.listdir(in_path)
    txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']

    for txt in txts: # img: img's path imgs: list of all img's path
        frame_str = txt.split('_')[-1].split('.')[0] 
        frame_name = txt.split('.')[0]
        frame_number = int(frame_str)   #每个seq中 隔10个txt输出一个

        if frame_number%10 == 0:
            in_path_txt = in_path + '/' + txt
            out_path_txt = out_path + '/'+ frame_name  + '.txt' #输出文件的路径
            shutil.copy(in_path_txt,out_path_txt) # 复制
            print(out_path_txt +" is done")
    
#in_path = '/userhome/viped/imgs/train'   #输入文件夹路径
#out_path = '/userhome/viped/imgs-low/train'  #输出文件夹路径

in_path = '/userhome/viped/bbs/val'
out_path = '/userhome/viped/labels/val'

txt_filter(in_path,out_path)
print("end")