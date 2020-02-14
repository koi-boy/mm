import os, json
import xml.etree.ElementTree as ET
import random
random.seed(12345)


def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' %(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' %(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

def convert(name_list, xml_dir, save_json):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    bnd_id = 1
    image_id = 1
    for name in name_list:
        name = name.strip()
        image = {'file_name': 'imgs_'+name+'_0.jpg', 'height': 3000, 'width': 4096, 'id': image_id}
        json_dict['images'].append(image)

        for i in range(5):
            full_name = 'imgs_' + name + '_' + str(i) + '.xml'
            xml_f = os.path.join(xml_dir, full_name)
            try:
                tree = ET.parse(xml_f)
                root = tree.getroot()
                for obj in get(root, 'object'):
                    category_id = i
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = float(get_and_check(bndbox, 'xmin', 1).text)
                    ymin = float(get_and_check(bndbox, 'ymin', 1).text)
                    xmax = float(get_and_check(bndbox, 'xmax', 1).text)
                    ymax = float(get_and_check(bndbox, 'ymax', 1).text)
                    assert(xmax > xmin)
                    assert(ymax > ymin)
                    o_width = abs(xmax - xmin)
                    o_height = abs(ymax - ymin)
                    ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id': image_id,
                           'bbox': [xmin, ymin, o_width, o_height],
                           'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                           'segmentation': []}
                    json_dict['annotations'].append(ann)
                    bnd_id += 1
            except:
                continue
        image_id += 1

    for i in range(5):
        cat = {'supercategory': 'none', 'id': i, 'name': '11_{}'.format(i)}
        json_dict['categories'].append(cat)

    json_fp = open(save_json, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    root = '/Users/dyy/Desktop/chongqing/round2'
    jiuye = os.listdir(os.path.join(root, 'jiuye'))
    names = []
    for img in jiuye:
        name = img.split('_')[1]
        if name not in names:
            names.append(name)

    train = random.sample(names, int(len(names) * 0.8))
    val = set(names) - set(train)

    xml_dir = os.path.join(root, 'defect_xmls')
    convert(train, xml_dir, os.path.join(root, 'jy_train.json'))
    convert(val, xml_dir, os.path.join(root, 'jy_val.json'))


