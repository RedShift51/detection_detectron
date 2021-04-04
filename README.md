# detection_task
The main goal of the task is to provide algorithm,
which is responsible to detect and to recognize diseases on x-ray images. For the simplicity I merged all classes into one just to detect a desease.
Such tasks in computer vision is preferred to solve with CNN (convolutional neural networks) 
with object detection outputs. 
Initially we have ~900 x-ray images, random 95% part of them is for training, another part is for testing.  
Further I will provide examples on several images for both parts. 

How to estimate quality of detections obtained by our algorithm? We have Intersection over Union (IoU) metric,
which can be used to evaluate accuracy of bounding boxes sizes. For 2 rectangles (bounding boxes) IoU is equal to:
![image](https://user-images.githubusercontent.com/29106459/113515345-a7491500-957c-11eb-8e45-eecc757636e7.png)

For the training part IoU = 0.6, for testing part IoU = 0.35.
Histograms are represented below.
![train_iou](https://user-images.githubusercontent.com/29106459/113516547-8506c580-9583-11eb-840c-8773531bbfea.png)
![test_iou](https://user-images.githubusercontent.com/29106459/113516549-8637f280-9583-11eb-9785-802bfe132a48.png)

As you can see on histograms, some objects on train/test stages are not recognized, this happened due to
insufficient manual tuning of parameters related to test. But objects on images are relatively small, and 
such results show us, that detection of deseases (mostly) is accurate.
Examples of detections on training and testing parts of dataset are presented below.

Examples of inference on train images:

![3](https://user-images.githubusercontent.com/29106459/113507007-9da9b800-9550-11eb-91c9-60ba20dab9c6.png)
![35](https://user-images.githubusercontent.com/29106459/113507035-d6499180-9550-11eb-9baf-6563f3c2d25b.png)
![8](https://user-images.githubusercontent.com/29106459/113507042-dfd2f980-9550-11eb-91ad-f9f579c62285.png)
![13](https://user-images.githubusercontent.com/29106459/113507046-e5304400-9550-11eb-904e-0f364b28398c.png)
![56](https://user-images.githubusercontent.com/29106459/113507056-efead900-9550-11eb-97fb-d9103c0b946e.png)
![22](https://user-images.githubusercontent.com/29106459/113507060-f6795080-9550-11eb-955b-724fbaa0454f.png)
![48](https://user-images.githubusercontent.com/29106459/113507065-fda05e80-9550-11eb-974e-7e75562430b8.png)
![38](https://user-images.githubusercontent.com/29106459/113507075-06913000-9551-11eb-949c-50273cd54c2c.png)
![35](https://user-images.githubusercontent.com/29106459/113507083-1a3c9680-9551-11eb-99d1-3fe7ef97adb5.png)
![41](https://user-images.githubusercontent.com/29106459/113507086-1f99e100-9551-11eb-9513-37437c08ea16.png)
![59](https://user-images.githubusercontent.com/29106459/113507090-232d6800-9551-11eb-904c-c7cb0485af8c.png)



On test images:
![3](https://user-images.githubusercontent.com/29106459/113507106-36d8ce80-9551-11eb-96f3-926620f42f86.png)
![7](https://user-images.githubusercontent.com/29106459/113507117-3fc9a000-9551-11eb-85ff-e3d67ae5afe0.png)
![6](https://user-images.githubusercontent.com/29106459/113507120-49530800-9551-11eb-99fd-5ea99dc69a70.png)
![36](https://user-images.githubusercontent.com/29106459/113507123-4ce68f00-9551-11eb-8c76-d757c5e751c3.png)
![37](https://user-images.githubusercontent.com/29106459/113507131-54a63380-9551-11eb-8587-2319e9a33415.png)
![26](https://user-images.githubusercontent.com/29106459/113507138-5a9c1480-9551-11eb-8a33-10e5fb5a89ed.png)
![40](https://user-images.githubusercontent.com/29106459/113507146-65ef4000-9551-11eb-8d5c-6e4d563f3e0b.png)
![29](https://user-images.githubusercontent.com/29106459/113507162-79021000-9551-11eb-94a9-95f55d7a0dc5.png)
![13](https://user-images.githubusercontent.com/29106459/113507169-87502c00-9551-11eb-8e2f-66ba3b3cd9b8.png)
![24](https://user-images.githubusercontent.com/29106459/113507175-946d1b00-9551-11eb-8b6b-91fd858f82ba.png)


All comparisons between predicted and GT (and trained model) you can find on 




=================================================


Trained few architectures (the best choice is resnet 50 + FPN + faster rcnn (16 roi to rpn, 1 class))


Few notes:
1) I have to tune NMS to reject few bounding boxes for one object in predictions in the same "grid cell".
2) I have to tune threshold for testing.

And I suppose, it is more than possible to train model for few classes.

Metrics during training
![image](https://user-images.githubusercontent.com/29106459/113508886-2a597380-955b-11eb-8cb3-9e8cb94921e6.png)
