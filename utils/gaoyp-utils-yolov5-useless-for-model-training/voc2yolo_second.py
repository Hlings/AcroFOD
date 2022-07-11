import shutil
import os
os.system('mkdir /userhome/VOC/')
os.system('mkdir /userhome/VOC/images')
os.system('mkdir /userhome/VOC/images/train')
os.system('mkdir /userhome/VOC/images/val')
os.system('mkdir /userhome/VOC/images/test')

os.system('mkdir /userhome/VOC/labels')
os.system('mkdir /userhome/VOC/labels/train')
os.system('mkdir /userhome/VOC/labels/val')
os.system('mkdir /userhome/VOC/labels/test')

print(os.path.exists('/userhome/2021_train.txt'))
f = open('/userhome/2021_train.txt', 'r')
lines = f.readlines()

for line in lines: # line /userhome/VOCdevkit/VOC2021/JPEGImages/000003.jpg
    line = "/"+"/".join(line.split('/')[-5:]).strip() #这样操作之后其实 line 和上面是完全一样的
    os.system("cp "+ line + " /userhome/VOC/images/train") #复制图片
    #shutil.copy(line,"/userhome/VOC/images/train")    
        
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')  
    #print("coping",line)
    os.system("cp "+ line + " /userhome/VOC/labels/train") #复制标签
    

print(os.path.exists('/userhome/2021_test.txt'))
f = open('/userhome/2021_test.txt', 'r')
lines = f.readlines()

for line in lines: # line /userhome/VOCdevkit/VOC2021/JPEGImages/000003.jpg
    line = "/"+"/".join(line.split('/')[-5:]).strip() #这样操作之后其实 line 和上面是完全一样的
    os.system("cp "+ line + " /userhome/VOC/images/test") #复制图片
        
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')  
    #print("coping",line)    
    os.system("cp "+ line + " /userhome/VOC/labels/test") #复制标签

print(os.path.exists('/userhome/2021_val.txt'))
f = open('/userhome/2021_val.txt', 'r')
lines = f.readlines()

for line in lines: # line /userhome/VOCdevkit/VOC2021/JPEGImages/000003.jpg
    line = "/"+"/".join(line.split('/')[-5:]).strip() #这样操作之后其实 line 和上面是完全一样的
    os.system("cp "+ line + " /userhome/VOC/images/val") #复制图片
        
    line = line.replace('JPEGImages', 'labels')
    line = line.replace('jpg', 'txt')  
    #print("coping",line)    
    os.system("cp "+ line + " /userhome/VOC/labels/val") #复制标签

