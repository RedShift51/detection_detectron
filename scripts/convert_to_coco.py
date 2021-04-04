import os
import numpy as np
import pandas as pd
import cv2
import json
import copy
import random

path_to_imgs = "/home/alex/tools/task/images_loc/"
json_path = "/home/alex/tools/task/markup.json"

with open("/home/alex/tools/task/train_val_list.txt", "r") as ftxt:
    trainval = set([k.strip() for k in ftxt.readlines()])

with open("/home/alex/tools/task/test_list.txt", "r") as ftxt:
    test = set([k.strip() for k in ftxt.readlines()])

txts = [trainval, test]

def main():
    # txt for mode: train, test, etc
    json_names = ["/home/alex/tools/task/train.json", \
                    "/home/alex/tools/task/valid.json"]
    # all images
    all_imgs = os.listdir(path_to_imgs)
    random.shuffle(all_imgs)
    train_imgs, val_imgs = copy.deepcopy(all_imgs[:int(len(all_imgs)*0.95)]), \
                        copy.deepcopy(all_imgs[int(len(all_imgs)*0.95):])
    list_imgs = [train_imgs, val_imgs]

    df = pd.read_csv("BBox_List_2017.csv")
    labels = {"obj": 0 }
    #{k: k0 for k0,k in enumerate(list(pd.unique(df["Finding Label"])))}

    bnd_id = 0
    for type_t in range(2):
        final_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
        imgs_paths = copy.deepcopy(list_imgs[type_t])
        for img0,img in enumerate(imgs_paths):
            image_id = img0
            img_mat = cv2.imread(os.path.join(path_to_imgs, img))
            height, width, _ = img_mat.shape
            image = copy.deepcopy({"file_name": img, "height": height, \
                                    "width": width, "id": image_id})
            final_dict["images"].append(image)
            #print(img[img.rfind("/")+1:])
            #print(df[df["Image Index"] == img[img.rfind("/")+1:]])

            markup = copy.deepcopy(df[df["Image Index"] == img[img.rfind("/")+1:]]).reset_index()
            #print(markup.shape)
            for obj_id in range(markup.shape[0]):
                xmin = max(int(markup["Bbox [x"][obj_id])-1, 0)
                xmax = int(markup["Bbox [x"][obj_id] + markup["w"][obj_id])
                ymin = max(int(markup["y"][obj_id])-1, 0)
                ymax = int(markup["y"][obj_id] + markup["h]"][obj_id])
                ann = {"area": markup["w"][obj_id] * markup["h]"][obj_id], \
                        "bbox": [xmin, ymin, xmax-xmin, ymax-ymin], \
                        "category_id": 0, "id": bnd_id,\
                        "ignore": 0, "segmentation": [], "iscrowd": 0, "image_id": image_id}
                #labels[str(markup["Finding Label"][obj_id])]
                final_dict["annotations"].append(copy.deepcopy(ann))
                bnd_id += 1
            print(img0)
        # labels / categories
        for cate, cid in labels.items():
            cat = {"supercatgory": "none", "id": cid, "name": cate}
            final_dict["categories"].append(copy.deepcopy(cat))

        with open(json_names[type_t], "w") as json_fp:
            json_str = json.dumps(final_dict)
            json_fp.write(json_str)



if __name__ == "__main__":
    main()
