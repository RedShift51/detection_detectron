import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

path_1 = "/home/alex/tools/task"
p_gt = "pic_valid"
p_out = "out_test"

names = os.listdir(os.path.join(path_1, p_gt))
random.shuffle(names)

for f0,f in enumerate(names[:60]):
    plt.subplot(121)
    img_gt = cv2.imread(os.path.join(path_1, p_gt, f))
    plt.title("gt")
    plt.imshow(img_gt)

    plt.subplot(122)
    img_out = cv2.imread(os.path.join(path_1, p_out, f))
    plt.title("output")
    plt.imshow(img_out)

    plt.savefig(os.path.join(path_1, "comparison_test", str(f0)+".png"))
    plt.close()
    print(f0)
