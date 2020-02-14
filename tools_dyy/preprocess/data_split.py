import os, json, shutil, glob
import pickle
import cv2
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return round(float(obj), 2)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

# root = '/Users/dyy/Desktop/chongqing/round2'
# img_dir = os.path.join(root, 'normal')
# imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
# save_dir = os.path.join(root, 'jiuye_normal')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# count = 0
# for img in imgs:
#     name = img.split('/')[-1]
#     if name.startswith('imgs'):
#         shutil.copy(img, os.path.join(save_dir, name))
#         count += 1
# print(count)

jiuye = os.listdir('/Users/dyy/Desktop/chongqing/round2/jiuye')
names = []
for img in jiuye:
    name = img.split('_')[1]
    if name not in names:
        names.append(name)

# images = glob.glob('/Users/dyy/Desktop/chongqing/round2/images/*.jpg')
# dst = '/Users/dyy/Desktop/jiuye'
# if not os.path.exists(dst):
#     os.makedirs(dst)
# for img in images:
#     name = img.split('/')[-1]
#     if name.split('_')[1] in names:
#         shutil.copy(img, os.path.join(dst, name))

img_dir = '/Users/dyy/Desktop/chongqing/round2/images'
img_dict = {}
mean, std = 0, 0
for i, name in enumerate(names):
    print(i, name)
    image = []
    for i in range(5):
        img_name = 'imgs_' + name + '_' + str(i) + '.jpg'
        img = cv2.imread(os.path.join(img_dir, img_name), 0)
        img = img[..., np.newaxis]
        image.append(img)
    concat = np.concatenate(image, axis=-1)
    mean += np.mean(concat)
    std += np.std(concat)
    img_dict[name] = concat
print(mean/len(names))
print(std/len(names))
# with open('/Users/dyy/Desktop/test.pkl', 'wb') as f:
#     pickle.dump(img_dict, f)

# with open('/Users/dyy/Desktop/jy.pkl', 'rb') as f:
#     data = pickle.load(f)
# print(data)

