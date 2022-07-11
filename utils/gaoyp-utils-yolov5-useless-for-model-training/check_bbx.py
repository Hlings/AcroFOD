import os
import xml.etree.ElementTree as ET 

def compare_min_max(xml_dir):
    xmls_old = os.listdir(xml_dir)
    xmls_old.sort()
    xmls = []
    for xml in xmls_old:
        if xml.split('.')[-1] == 'xml':
            xmls.append(xml)
    print('the length of xmls is', len(xmls))
    flag = 0
    count = 0
    for xml in xmls:
        xml_path = os.path.join(xml_dir, xml)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for img_size in root.findall('size'):
            width = img_size.find('width').text
            height= img_size.find('height').text
            
        for elem in root.findall('object'):
            xmin = elem.find('bndbox').find('xmin').text
            ymin = elem.find('bndbox').find('ymin').text
            xmax = elem.find('bndbox').find('xmax').text
            ymax = elem.find('bndbox').find('ymax').text
            if (int(xmin) == 0 and int(ymin) == 0) or (int(xmax) == 0 and int(ymax) == 0):
                print('min or max == 0',xml_path)
                flag = 1
            if int(xmin) < 0 or int(ymin) < 0:
                print(' min < 0 mistake', xml_path)
                flag = 1                
            if int(xmax) > int(width) or int(ymax) > int(height):
                print(' beyond border', xml_path)
                flag=1            
            if int(ymin) > int(ymax) or int(xmin) > int(xmax):
                print('min > max in file:',xml_path)
                flag = 1
        if flag == 1:
            count += 1
            flag = 0
        #`print('cur file is ',xml_path)
    print('{} files that min > max'.format(count))
    print('finish comparision...')

if __name__ == '__main__':
    xml_dir = '/userhome/voc_mini/voc_1/voc_1_1_voc/VOC2012/Annotations'
    compare_min_max(xml_dir)

    
