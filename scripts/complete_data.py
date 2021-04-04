import os

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches 
import cv2#_imshow# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import matplotlib.pyplot as plt
import time
import copy

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_valid", {}, \
                "/home/alex/tools/task/valid.json", "/home/alex/tools/task/images_loc/")


my_dataset_train_metadata = MetadataCatalog.get("my_dataset_valid")
dataset_dicts = DatasetCatalog.get("my_dataset_valid")
import random
from detectron2.utils.visualizer import Visualizer
for d in dataset_dicts:
    #random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    #plt.imshow(vis.get_image()[:, :, ::-1])

    temp_name = copy.deepcopy(d["file_name"])
    temp_name = temp_name[temp_name.rfind("/")+1:]
    cv2.imwrite("pic_valid/" + temp_name, vis.get_image()[:, :, ::-1])


