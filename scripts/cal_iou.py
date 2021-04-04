import os
import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

with open("box_test_gt.pkl", "rb") as f:
    gt = pickle.load(f)

with open("box_test_pred.pkl", "rb") as f:
    pred = pickle.load(f)

all_ious = []
for f0,f in enumerate(list(pred.keys())):
    curr_iou = 0
    for box in gt[f]:
        box_gt = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        for box_p in pred[f]:
            #print(box)
            #print(box_p)
            curr_iou = max(bb_intersection_over_union(box_gt, box_p), curr_iou)
    print("==============", f0, curr_iou)
    all_ious.append(copy.deepcopy(curr_iou))
    #if f0 > 5:
    #    1/0

plt.hist(all_ious, bins=20)
plt.grid()
plt.xlabel("IoU")
plt.title("test IoU histogram")
plt.show()
print(np.mean(all_ious))




