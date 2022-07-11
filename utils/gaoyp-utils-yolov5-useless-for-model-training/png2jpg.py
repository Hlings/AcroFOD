import cv2
import os

def png2lowjpg(in_path,out_path,new_dim):
    
    imgs = os.listdir(in_path)
    imgs = [img for img in imgs if img.split('.')[-1] == 'png']

    for img in imgs: # img: img's path imgs: list of all img's path
        frame_str = img.split('_')[-1].split('.')[0] 
        frame_name = img.split('.')[0]
        frame_number = int(frame_str)   #每个seq中 隔10张图片输出一个

        if frame_number%10 == 0:
            in_path_img = in_path + '/' + img
            src = cv2.imread(in_path_img)
            #im = cv2.resize(src, dsize=new_dim, interpolation=cv2.INTER_LINEAR)
            out = out_path + '/'+ frame_name  + '.jpg' #输出文件的路径
            cv2.imwrite(out,im)
            print(out+" is done")
    
in_path = '/userhome/delete/viped-o/imgs/train'#输入文件夹路径
out_path = '/userhome/viped/images/train_highr'#输出文件夹路径
new_dim = (640,360)

png2lowjpg(in_path,out_path,new_dim)
print("end")