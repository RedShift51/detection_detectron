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
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["LRU_CACHE_CAPACITY"] = "1"
os.environ["LD_PRELOAD"] = "./libjemalloc.so.1"

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, \
                "/home/alex/tools/task/train.json", "/home/alex/tools/task/images_loc/")
#register_coco_instances("my_dataset_valid", {}, \
#                "/home/alex/tools/task/valid.json", "/home/alex/tools/task/images_loc/")


cfg = get_cfg()
#cfg.merge_from_file(\
#    "/home/alex/tools/task/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
#cfg.merge_from_file(\
#    "/home/alex/tools/task/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml")
cfg.merge_from_file(\
    "/home/alex/tools/task/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

#"/home/alex/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()#("my_dataset_valid",)
#cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
#cfg.MODEL.WEIGHTS = \
#    "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#        model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.005
#cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.MAX_ITER = 50000 #adjust up if val mAP is still rising, adjust down if overfit
#cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.STEPS = (17000, 28000, 39000)
cfg.SOLVER.GAMMA = 0.15
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16 #64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1#8
cfg.TEST.EVAL_PERIOD = 500
#cfg.MODEL.WEIGHTS = os.path.join("/home/alex/tools/task/output", "model_final.pth")
cfg.FILTER_EMPTY_ANNOTATIONS = 1

num_gpu = 1.
bs = num_gpu
#cfg.SOLVER.BASE_LR = 0.02 * bs / 16.  # pick a good LR
cfg.WORLD_SIZE = 1

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

#from detectron2.data.datasets import register_coco_instances
#register_coco_instances("my_dataset_train", {}, \
#                "/home/alex/tools/task/trainval.json", "/home/alex/tools/task/images_loc/")

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

