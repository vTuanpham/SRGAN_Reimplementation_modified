
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


train_dir = "train_dir"
ix = 0
for img in os.listdir(train_dir +"/original_images/Flickr1024/Train/L"):
    ix += 1
    img_array = cv.imread(train_dir + "/original_images/Flickr1024/Train/L/" + img)

    img_array = cv.resize(img_array, (1024,1024))
    # lr_img_array = cv.GaussianBlur(img_array,(5,5),0)
    lr_img_array = cv.resize(img_array,(256,256))
    cv.imwrite(train_dir+"/hr_images1024/"+'im'+str(int(ix))+'.jpg',img_array)
    cv.imwrite(train_dir+"/lr_images256/"+'im'+str(int(ix))+'.jpg',lr_img_array)