# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import imageio
import argparse
import datetime
# utils
# __________________________________________________
import logging
import math
import os
import subprocess
from pathlib import Path

import cv2
import keras as keras
import numpy as np
import torch

import Segmentation3
from Unet_model import UNET
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from yolov5.models.common import DetectMultiBackend

typpe=["ford focus", "honda civic","kia picanto","toyota corolla"]
numero=[0]

# from albumentations.pytorch.transforms import ToTensorV2
def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def git_describe(path=Path(__file__).parent):  # path must be a directory
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    s = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(s, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError as e:
        return ''  # not a git repository

def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-3-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'



#end utils
#____________________________________________________________
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = select_device(device)
YOLO =DetectMultiBackend(weights='yolov5s.pt', device=DEVICE, dnn=False)
Unet=UNET(in_channels=3, out_channels=1).to(DEVICE)
Unet.load_state_dict(torch.load("my_checkpoint.pth.tar",
                             map_location=torch.device(DEVICE))["state_dict"])  # remove map_lcoation when using gpu
Unet.eval()
TYPE= keras.models.load_model('saved_model')


def find_car(car): #find if there is a car in the image at all


    img0 = cv2.imread(car)
    imgsz = check_img_size([480,480], s=32)  # check image size img0.shape[0:2]
    img = letterbox(img0, imgsz, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE)
    img=img.float()
    # img /= 255
    img=img[None]

    #TODO: Crop the car out of the image with YOLO built in crop feature
    pred =  non_max_suppression(YOLO(img))
    for i, det in enumerate(pred):
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()
            if YOLO.names[int(c)] == 'car' and n>0:
                return True

    return False #yolo5

def car_type(img): #predict type of car, from ["ford focus", "honda civic","kia picanto","toyota corolla"]
    # image = np.expand_dims(cv2.resize(img, (480, 480)), axis=0)
    pred=TYPE.predict(np.expand_dims(cv2.resize(img, (480, 480)), axis=0)) #["ford focus", "honda civic","kia picanto","toyota corolla"]

    return np.where(pred[0]==np.max(pred[0]))[0]


def cut_and_segment(car,sizeX=480,sizeY=480): #Segment car from image
    img = Segmentation3.run_segmentationn(car)
    nonzero = np.nonzero(img)
    Xmin, Xmax = nonzero[1].min(), nonzero[1].max()
    Ymin, Ymax = nonzero[0].min(), nonzero[0].max()
    midX = int(round(Xmax + Xmin) / 2)
    img = img[Ymin:Ymax, Xmin:Xmax]
    img = cv2.resize(img, (sizeX, sizeY))
    return img # segmentation3

def find_mask(img_): #predict anomaly mask
    img = letterbox(img_, (160, 160), stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE)
    img = img.float()
    img /= 255
    img = img[None]
    with torch.no_grad():
        preds = torch.sigmoid(Unet(img))
        preds = (preds > 0.5).float()
    return preds #unet

def color_test(car1_image,car2_image): #find avarage of color of the cars, return True if both colors are close enough
    col_range=32
    average_color1=np.average(car1_image[120:360,120:360],(0,1))
    average_color2 = np.average(car2_image[120:360, 120:360], (0, 1))

    return (average_color1+col_range>average_color2).all() and (average_color2>average_color1-col_range).all()


def compare_masks_full(mask1,mask2,success=0.1,cpuComponent = True): #compare anomaly masks, return True if their success
    if cpuComponent:
        m1,m2=np.array(mask1.cpu(),dtype='uint8')[0][0],np.array(mask2.cpu(),dtype='uint8')[0][0]
    else:
        m1, m2 = np.array(mask1[0][0].cpu(), dtype='uint8'), np.array(mask2, dtype='uint8')
    k= np.count_nonzero(np.logical_or(m1,m2))
    if k < 20:
        return True

    j1 = np.count_nonzero(np.logical_and(m1, m2))
    m3=np.flip(m1)                                  #flip, if car pictures were flipped before
    j2 = np.count_nonzero(np.logical_and(m3, m2))
    M1,M2 = cv2.moments(m1),cv2.moments(m2)         #centerlize masks, overlay over each other
    if M1["m00"] == 0 or M2["m00"] ==0:
        j3=0
    else:
        cX1,cX2 = int(M1["m10"] / M1["m00"]),int(M2["m10"] / M2["m00"])
        cY1,cY2 = int(M1["m01"] / M1["m00"]),int(M2["m01"] / M2["m00"])
        M3 =  np.float32([[1, 0, cY1-cY2], [0, 1, cX1-cX2]])
        m1 = cv2.warpAffine(m1, M3, m1.shape)
        # imageio.imwrite('img/temp/6.jpg', 85 * m1 + 170 * m2)
        j3 = np.count_nonzero(np.logical_and(m1, m2))
    showMasks(m1,m2,max(j1/k,j2/k,j3/k) >success)
    return max(j1/k,j2/k,j3/k) >success,j1/k,j2/k,j3/k

def compare_masks_cut(masks1,masks2,success=0.1):  #compare anomaly masks,
    # by parts (cut into 4 parts then compare each part) return True
    # if their success is lower than
    maxes = 0
    p=0
    for i in range (4):
        m1,m2 = masks1[i],masks2[i]
        k= np.count_nonzero(np.logical_or(m1,m2))
        if k < 20:
            continue
        j1=np.count_nonzero(np.logical_and(m1,m2))
        m1=np.flip(m1)
        j2=np.count_nonzero(np.logical_and(m1,m2))
        if maxes < max(j1 / k, j2 / k):
            maxes = max(j1 / k,j2 / k)
            p=i
    # if maxes<success:
    #     for i in range(4):
    #         imageio.imwrite('img/temp/'+str(i)+'.jpg',2*masks1[i]+masks2[i])
    return maxes>success,maxes,p

def prediction_main(car_path1,car_path2): #main prediction function
    #returns True if all tests return True

    car1_auth = find_car(car_path1)
    car2_auth = find_car(car_path2)
    if not car1_auth:
        print("there is no car in image 1")
        return False
    elif not car2_auth:
        print("there is no car in image 2")
        return False
    car1_image = cut_and_segment(car_path1)
    car2_image = cut_and_segment(car_path2)
    type1=car_type(car1_image)
    type2 = car_type(car2_image)

    if type1 != type2:
        # print("not the same car type")
        return False
    if not color_test(car1_image,car2_image):
        # print("wrong color")
        return False
    mask1 = find_mask(car1_image)
    mask2 = find_mask(car2_image)
    maskk1 = np.zeros((4,160,160))
    maskk2 = np.zeros((4, 160, 160))
    for i in range(2):
        for j in range(2):
            maskk1[i*2+j] = np.array(find_mask(car1_image[i*240:i*240+240,j*240:j*240+240]).cpu()[0][0],dtype='uint8')
            maskk2[i*2+j] = np.array(find_mask(car2_image[i*240:i*240+240,j*240:j*240+240]).cpu()[0][0],dtype='uint8')

    comp=compare_masks_cut(maskk1,maskk2,0.095)#0.045 gives 85% success, 0.1 gives 75%
    com = compare_masks_full(mask1,mask2,0.095)
    if com[0] or comp[0]:
        print("comparison good:" + str(com) + str(comp))
        #save or show the images of both cars and their anomalies, as the false negatives are too high
        return True
    else:
        print("comparison bad:" + str(com[1])+" , " + str(com[2]) + " , " + str(comp[1]))
        return False

def showMasks(mask1,mask2,TF):
    mask1=mask1*150
    mask2=mask2*100
    mask = mask1+mask2

    cv2.imwrite("/home/danielalon/compare/imgs/comparing masks/"+str(numero[0])+str(TF)+".jpg",mask)
    numero[0]=numero[0]+1

def prediction_util(dir,is_same):
    #returns the number of predictions of bool type is_same from all the files in dir
    s=0
    p=0
    for dirr in os.scandir(dir):
        if not os.path.isdir(dirr):
            continue
        i = os.scandir(dirr)
        a,b = next(i),next(i)
        if not (a.name[-3:] == 'png' or a.name[-3:]=='jpg' or b.name[-3:] == 'png' or b.name[-3:]=='jpg' or b.name[-3:]=='jpeg' or a.name[-3:]=='jpeg'):
            print("one of the files in the directory \n" + dirr.path + "\nis not a picture, please provide pictures with ending .jpg or png ")
            continue

        if prediction_main(a.path,b.path)==is_same:
            s+=1
        p+=1
    return s,p

def make_dir_mask(directory,mask,masks):
    os.mkdir(directory)
    cv2.imwrite(directory + "/0.jpg", np.array(mask[0][0].cpu()))
    for i in range(0, 4):
        cv2.imwrite(directory + "/" + str(i+1)+".jpg", masks[i])

def make_color_dir(directory,mask,masks):

    os.mkdir(directory)
    directory = directory + "/" + "0"
    make_dir_mask(directory, mask, masks)


def comparison(img,img_dir): # there is also a simple way to do that with just names, but directories seem to me  more elegant
    base =64
    car1_auth = find_car(img)
    if not car1_auth:
        print("no car in image")
        return False
    try:
        car_image = cut_and_segment(img)
    except:
        print("car not recognized, try again")
        return False

    type = car_type(car_image)
    mask = find_mask(car_image)
    color = np.average(car_image[120:360,120:360],(0,1))
    color = base * np.round(color/base,0,)
    color = str((int(color[0]),int(color[1]),int(color[2])))[1:-1]
    maskk = np.zeros((4, 160, 160))
    for i in range(2):
        for j in range(2):
            maskk[i * 2 + j] = np.array(
                find_mask(car_image[i * 240:i * 240 + 240, j * 240:j * 240 + 240]).cpu()[0][0], dtype='uint8')
    directory = img_dir+"/"+typpe[type[0]]
    if not os.path.isdir(directory):
        os.mkdir(directory)
        directory = directory+"/"+str(color)
        make_color_dir(directory,mask,maskk)
        return False
    directory = directory + "/" + str(color)
    if not os.path.isdir(directory):
        make_color_dir(directory, mask, maskk)
        return False
    for Dir in os.listdir(directory):
        mask2 = cv2.imread(directory+"/"+Dir+"/0.jpg",0)
        masks2 = [cv2.imread(directory+"/"+Dir+"/"+str(i)+".jpg",0) for i in range(0,4)]
        comp = compare_masks_cut(maskk, masks2, 0.089)
        com = compare_masks_full(mask, mask2, 0.089,False)
        if com or comp[0]:
            return True
    i = len(os.listdir(directory))
    make_dir_mask(directory+"/"+str(i),mask,maskk)

    return False

def comparison_test(img_directory,img_dictionary):
    for image in os.listdir(img_directory):
        comparison(img_directory+"/"+image,img_dictionary)




def prediction_test(dir):
    #main test, prints successes precentages

    same_dir = dir+"/same"
    different_dir = dir + "/not same"
    same_success,same_pairs=prediction_util(same_dir,True)
    print("__________________________________________________________________")
    dif_success,dif_pairs=prediction_util(different_dir, False)
    print("same success precentage:" + str((same_success) / (same_pairs) * 100))
    print("differenct success precentage:" + str((dif_success) / ( dif_pairs) * 100))
    print("success precentage:" + str((same_success+dif_success)/(same_pairs+dif_pairs)*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('car_compare')
    parser.add_argument('--mode', default='single', type=str,choices =['single','many','comparison','comparison_test' ], help='select mode, single for two images or many for a directory of same and not same cars')
    parser.add_argument('--image_path1', default=None, type=str, help='path to image1')
    parser.add_argument('--image_path2', default=None, type=str, help='path to image2')
    parser.add_argument('--dir_path', default=None, type=str, help='path to images directory, with sub directories: same, not same')
    parser.add_argument('--img_dictionary', default=None, type=str,
                        help='path to images dictionary')
    parser.add_argument('--imgs_in_path', default=None, type=str,
                        help='path to images directory, with many images to compare')
    args = parser.parse_args()
    if args.mode == 'single':
        if(prediction_main(args.image_path1, args.image_path2)):
            print("same car")
        else:
            print("not same car")
    elif args.mode == 'many':
        prediction_test(args.dir_path)
    elif args.mode == 'comparison':
        if (comparison(args.image_path1,args.img_dictionary)):
            print("car known")
        else:
            print("car not known")
    elif args.mode == 'comparison_test':
        comparison_test(args.imgs_in_path,args.img_dictionary)




