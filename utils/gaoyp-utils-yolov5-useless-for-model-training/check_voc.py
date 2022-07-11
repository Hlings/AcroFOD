import os

root_path = '/userhome/viped_voc/VOC2012'

check_set = os.path.join(root_path, 'ImageSets', 'Main')
check_txt = check_set + '/' + 'val.txt'

checked_txt = check_set + '/' + 'val_checked.txt' 

f = open(check_txt, 'r')
f2 = open(checked_txt, 'a+') 

number = 0
line = f.readline()
while line:
    line = line.replace('\n', '')
    xml_file = os.path.join(root_path, 'Annotations', (str(line) + '.xml'))  
    if os.path.exists(xml_file):
        f2.write(line + '\n')
    if not os.path.exists(xml_file):
        number += 1
        print(xml_file)
    line = f.readline()
        
f.close()
f2.close()
print("number of del file is", number)
    
