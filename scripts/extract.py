import os
import pandas as pd
import copy
import time

df = pd.read_csv("BBox_List_2017.csv")
names = set(list(df["Image Index"]))

files = [os.path.join("/home/alex/Downloads", k) for k in \
            os.listdir("/home/alex/Downloads") if \
            k.find("images_")!=-1 and k.find(".tar.gz")!=-1]

temp_path = "/home/alex/tools/task/temp"
dest_path = "/home/alex/tools/task/images_loc"
for f_name in files:
    #os.mkdir(temp_path)
    os.system("tar -C " + temp_path + " -zxvf " + f_name)
    time.sleep(5)

    curr_imgs = set(os.listdir(temp_path + "/images"))
    desired_img = copy.deepcopy(list(curr_imgs.intersection(names)))
    for im in desired_img:
        os.system("cp " + temp_path + "/images/" + im + " " + dest_path)
    
    os.system("rm -rf /home/alex/tools/task/temp/images")
    time.sleep(5)
