import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
#from detectron2.data import DatasetCatalog
#from detectron2.data import MetadataCatalog, 
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer
#from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

from torchvision import transforms
from torchvision.datasets import ImageFolder

import glob

from detectron2.data.datasets import register_coco_instances

import cv2

from detectron2.utils.logger import setup_logger
setup_logger()
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import torch
import json
from detectron2.checkpoint import DetectionCheckpointer

with open("/home/alex/tools/task/valid.json", "r") as f:
    data = json.load(f)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["LD_PRELOAD"] = "./libjemalloc.so.1"


cfg = get_cfg()
cfg.MODEL.WEIGHTS = "/home/alex/tools/task/output/model_0039999.pth"
cfg.merge_from_file(\
    "/home/alex/tools/task/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#model = build_model(cfg)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.77

#predictor = DefaultPredictor(cfg)
checkpoint = torch.load("/home/alex/tools/task/output/model_0039999.pth", \
                map_location=lambda storage, loc: storage)
#print(checkpoint.keys())
#1/0
model = build_model(cfg)
model.load_state_dict(checkpoint["model"])
model.eval()


for el0, el in enumerate(data["images"]):
    print(el["file_name"])

    im = cv2.imread(os.path.join("/home/alex/tools/task/images_loc", el["file_name"]))
    im_c = torch.from_numpy(im)
    outputs = model([{"image": im_c.permute(2, 0, 1)}])
    #predictor
    print(outputs)

    scores = outputs[0]["instances"].scores.to("cpu").detach().numpy()
    preds = outputs[0]["instances"].pred_boxes
    for bb in preds:
        bbcpu = bb.to("cpu").detach().numpy()
        cv2.rectangle(im, (bbcpu[0], bbcpu[1]), (bbcpu[2], bbcpu[3]), (0, 0, 255), 5)
    
    #img = v.get_image()[:, :, ::-1]
    cv2.imwrite("out_test/" + str(el["file_name"]), im)#g)
    print(el0)
    
