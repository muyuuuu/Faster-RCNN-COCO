'''
File: refactor-anno.py
Project: Faster-RCNN-COCO
File Created: Thursday, 6th May 2021 3:13:55 pm
Author: lanling (https://github.com/muyuuuu)
-----------
Last Modified: Thursday, 6th May 2021 3:38:04 pm
Modified By: lanling (https://github.com/muyuuuu)
Copyright 2020 - 2021 XDU, XDU
-----------
Description: 为了 torch 的 batch size 能对齐，重新生成 json
'''

import json
from pycocotools.coco import COCO
from tqdm import tqdm

anno_path = 'data/openbrand_train.json'
coco = COCO(anno_path)
ids = list(coco.imgs.keys())
image_dir = 'data/train_data/'
result = []

for idx in tqdm(range(len(ids))):
    img_id = ids[idx]

    ann_ids = coco.getAnnIds(imgIds=img_id)
    target = coco.loadAnns(ann_ids)

    img_ids = coco.getImgIds(imgIds=img_id)
    image = coco.loadImgs(img_ids)

    for i in target:
        d = {}
        for key in i.keys():
            if key == 'iscrowd':
                d['iscrowd'] = i[key]
            elif key == 'image_id':
                d['image_id'] = i[key]
            elif key == 'bbox':
                d['boxes'] = [
                    i[key][0], i[key][1], i[key][0] + i[key][2],
                    i[key][1] + i[key][3]
                ]
            elif key == 'category_id':
                d['labels'] = i[key]
            elif key == 'area':
                d['area'] = i[key]
            elif key == 'image_id':
                d['image_id'] = i[key]
        for j in image:
            for key in j.keys():
                if key == 'id':
                    assert j[key] == d['image_id']
                elif key == 'width':
                    d['width'] = j[key]
                elif key == 'height':
                    d['height'] = j[key]
                elif key == 'file_name':
                    d['file_name'] = j[key]
        result.append(d)

with open('reconstruct-anno.json', 'w') as f:
    json.dump(result, f, indent=4)