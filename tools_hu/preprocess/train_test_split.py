import shutil
import os
import random
import glob

def train_test_split(img_dir, xml_dir, train_size, output):
    imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
    train_set = random.sample(imgs, int(len(imgs)*train_size))
    train_dir = os.path.join(output, 'train')
    test_dir = os.path.join(output, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for img in imgs:
        img_name = img.replace('\\', '/').split('/')[-1]
        xml = img_name[:-4] + '.xml'
        if img in train_set:
            shutil.copy(img, os.path.join(train_dir, img_name))
            shutil.copy(os.path.join(xml_dir, xml), os.path.join(train_dir, xml))
        else:
            shutil.copy(img, os.path.join(test_dir, img_name))
            shutil.copy(os.path.join(xml_dir, xml), os.path.join(test_dir, xml))

if __name__ == '__main__':
    img_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\bottom\images'
    anns_dir = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\bottom\images'
    TRAIN_SIZE = 0.9
    output = r'D:\Project\chongqing_contest\data\chongqing1_round1_train1_20191223\bottom'
    train_test_split(img_dir, anns_dir, TRAIN_SIZE, output)
