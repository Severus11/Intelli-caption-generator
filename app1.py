# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:09:13 2021

@author: parth
"""
import tensorflow as tf
import matplotlib.pyplot as plt 

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image

annotation_file = 'annotations/captions_train2014.json'
image_folder ='train2014'

with open(annotation_file, 'r')as f:
    annotations = json.load(f)
    
image_path_caption = collections.defaultdict(list) 
for val in annotations['annotations']:
    caption = f"<start> {val['caption']} <end>"
    image_path = image_folder + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

train_image_paths = image_paths[:15000]
print(len(train_image_paths))