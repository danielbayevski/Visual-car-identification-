import os
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
# from keras.optimizer_v1 import Adam
from torchvision import transforms as pth_transforms
import math
import matplotlib.pyplot as plt

import Segmentation3
# import vision_transformer as vits
import tensorflow as tf

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from tensorflow import keras
import torch.backends.cudnn as cudnn

#
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras.optimizers as op
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import concatenate



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative






def segment(dir,output_dir):
    for i in os.listdir(dir):
        name = i
        if os.path.isdir(output_dir+"/"+name):
            pass
        else:
            os.mkdir(output_dir+"/"+name)
            continue
        Segmentation3.run_segmentation((dir+"/"+name,output_dir+"/"+name))

def augament(dir): # directory of image directories -> 480x480 image of the car blurred,grayscaled,flipped
    for i in os.listdir(dir):
        for j in os.scandir(dir+"/" + i):
            img = cv2.imread(j.path)
            nonzero = np.nonzero(img)
            Xmin, Xmax = nonzero[1].min(), nonzero[1].max()
            Ymin, Ymax = nonzero[0].min(), nonzero[0].max()
            midX = int(round(Xmax + Xmin) / 2)

            img = img[Ymin:Ymax, Xmin:Xmax]

            img = cv2.resize(img, (480, 480))
            cv2.imwrite(j.path, img)
            blur_image= cv2.blur(img,(10,10))
            img = cv2.flip(img, 1)
            cv2.imwrite(j.path[:-4]+"0"+j.path[-4:], img)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(j.path[:-4]+"1"+j.path[-4:], img)
            img = cv2.flip(img, 1)
            cv2.imwrite(j.path[:-4]+"2"+j.path[-4:], img)

            cv2.imwrite(j.path[:-4]+"3"+j.path[-4:], blur_image)
            img = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(j.path[:-4]+"4"+j.path[-4:], img)
            img = cv2.flip(blur_image, 1)
            cv2.imwrite(j.path[:-4]+"5"+j.path[-4:], img)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('train the car type recognizer')
    parser.add_argument("--dir_path", default=None, type=str, help="Path of the directory of images to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')

    parser.add_argument('--load_model', defalut=None, type = str, help="path to model, if have a previous model")
    parser.add_argument('--segment', default = False,type = bool, help = "segmentise the picture/batch, True/Flase")
    parser.add_argument('--crop_size', default=32,type=int,choices =[8,10,12,16,24,30,32,48],
                            help='the size of the crop of the image, current sizes are: 8,10,12,16,24,30,32 ' )
    parser.add_argument('--epoch_num', default = 2, type = int, help = "number of epochs to run")
    args = parser.parse_args()

    if args.segemnt: #to segment or not to segment.
        segment(args.dir_path,args.output_dir)

        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # print("num of GPU available:" + physical_devices)
        # tf.config.experimental.set_memory_growth(physical_devices[0],True)

    # model start
    model = keras.models.load_model('saved_model')
    clss=["ford focus", "honda civic","kia picanto","toyota corolla"]
    num_output_classes = 4  # 0 = ford focus, 1 = honda civic, 2 = kia picanto, 3= toyota corolla
    input_img_size = (480, 480,1)  # 64x64 image with 1 color channel

    if args.load_model == None:
        model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_img_size),
        MaxPooling2D(pool_size=(2, 2),strides=2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(num_output_classes, activation="softmax")
        ])
    #model end
    else:
        model = keras.models.load_model('saved_model')
    #data start
    train_path = args.dir_path+"/train"
    # test_path = args.dir_path+"/test"
    valid_path = args.dir_path+"/valid"

    train_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=train_path,target_size=(480,480), classes=clss, batch_size=10)
    valid_batches = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=valid_path, target_size=(480,480), classes=clss, batch_size=10)
    # test_batches = ImageDataGenerator(
    #     preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    #     directory=test_path, target_size=(480,480), classes=clss, batch_size=10)
    #data end
    # imgs,labels=next(train_batches)
    # plotimages(imgs)
    # print(labels)
    if args.load_model == None:
        model.summary()
        model.compile(optimizer=op.adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_batches,validation_data=valid_batches,epochs=args.epoch_num,verbose=2)



    keras.models.save_model(model,"saved_model")




