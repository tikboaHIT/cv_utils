import _pickle as pickle
from pycocotools.mask import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
from tqdm import tqdm

test_json_path = "/home/zhangyun/mmdetection-htc/data/coco/annotations/instances_test2014.json"
with open(test_json_path, "r") as f:
    test_str = json.load(f)

images = test_str["images"]

pkl_path = "/home/zhangyun/mmdetection-htc/tools/work_dirs/mask_rcnn_r50_fpn_1x/results.pkl"
inf = pickle.load(open(pkl_path, "rb"))

for sig_reuslt,image in zip(inf, images):
    image_name = image["file_name"]
    image_path = "/home/zhangyun/mmdetection-htc/data/coco/test2014/" + image_name
    img = plt.imread(image_path)
    img = (np.array(img) * 255).astype(np.uint8)

    bboxs_prob = sig_reuslt[0][0]
    segs = sig_reuslt[1][0]
    for seg, bbox in zip(segs, bboxs_prob):
        prob = bbox[4]
        if prob > 0.7:
            color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            mask = decode(seg).astype(bool)
            img[mask] = img[mask] * 0.6 + color_mask * 0.4

    plt.imshow(img)
    plt.show()
