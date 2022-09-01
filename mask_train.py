import os
import random
import shutil

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

import Segmentation3
from dataset import CarvanaDataset
from model import UNET
from utilss import check_accuracy
from utilss import get_loaders
from utilss import load_checkpoint
from utilss import save_checkpoint
from utilss import save_predictions_as_imgs

# Hyperparameters etc.
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100
NUM_EPOCHS = 7
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 160  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
car_path = "img/cars/"
masks_path = "img/mask_filters/"
scratch_path = "img/scratch_mask/"
out_path= "img/scratched_image_binary/"
mask_out_path = "img/bimaps/"
TRAIN_IMG_DIR = "img/scratched_image_binary/"
TRAIN_MASK_DIR = "img/bimaps/"
VAL_IMG_DIR = "img/im_val/"
VAL_MASK_DIR = "img/bi_val/"


def whiten():
    # for i in os.scandir(TRAIN_MASK_DIR):
    #     img = cv2.imread(i.path,0)
    #     img = cv2.resize(img,(480,480))
    #     if img.max()>1:
    #         img = img/255
    #     cv2.imwrite(i.path,img)
    for i in os.scandir(VAL_IMG_DIR):
        img = cv2.imread(i.path, 0)
        img = cv2.resize(img, (480, 480))
        cv2.imwrite(i.path, img)
    # for i in os.scandir(VAL_MASK_DIR):
    #     img = cv2.imread(i.path,0)
    #     img = cv2.resize(img,(480,480))
    #     if img.max()>1:
    #         img = img/255
    #     cv2.imwrite(i.path,img)
    for i in os.scandir(TRAIN_IMG_DIR):
        img = cv2.imread(i.path, 0)
        img = cv2.resize(img, (480, 480))
        cv2.imwrite(i.path, img)

def to255(im_dir):
    for i in os.scandir(im_dir):
        img = cv2.imread(i.path, 0)
        img = img*255
        cv2.imwrite(i.path, img)
def from255(im_dir):
    for i in os.scandir(im_dir):
        img = cv2.imread(i.path, 0)
        img = img/255
        cv2.imwrite(i.path, img)


def add_scratches():
    cars = os.scandir(car_path)
    # scratches2 = os.listdir(scratch_path)
    for car in cars:
        img=cv2.imread(car.path,0)
        masks = os.scandir(masks_path)
        car_mask = img>0
        for maskk in masks:
            scratches = os.scandir(scratch_path)

            for scrach in scratches:
                r = random.randint(1, 12)

                if r >7:
                    continue

                name = out_path + car.name[:-4] + maskk.name[:-4] + scrach.name[:-4] + ".jpg"
                name1 = mask_out_path + car.name[:-4] + maskk.name[:-4] + scrach.name[:-4] + ".jpg"


                scratch = np.round(
                    cv2.imread(scrach.path,0) / 255)
                mask = cv2.imread(maskk.path,0)
                y_rand=random.randint(-100,100)
                x_rand=random.randint(-100,100)
                M = np.float32([[1, 0, y_rand], [0, 1, x_rand]])
                mask = cv2.warpAffine(img, M, img.shape) * car_mask
                if r%4==0:
                    mask = cv2 .flip(mask,1)
                if r % 3 == 0:
                    scratch = cv2.flip(scratch, 1)
                if r%6==0:
                    r2=random.randint(0,14)
                    scr_path = os.listdir(scratch_path)[r2]
                    scratch2_ = np.round(cv2.imread(scratch_path+"/"+scr_path,0) / 255)
                    scratch = cv2.bitwise_or(scratch, scratch2_)
                if r%7==0:
                    scratch = np.zeros(scratch.shape)

                image = np.array(img - (img * scratch) + scratch * mask,dtype='uint8')
                image = cv2.blur(image,(3,3))

                cv2.imwrite(name,cv2.resize(image,(160,160)))
                scratch = np.array(scratch, dtype='uint8')


                # imggg = open_morph(open_morph(scratch))

                #new_mask = 1+cv2.dilate(open_morph(open_morph(open_morph(scratch))),(3,3))-2*scratch
                new_mask = scratch

                cv2.imwrite(name1,cv2.resize(np.array(new_mask, dtype='uint8'),(160,160)))
                for i in range(2):
                    for j in range(2):
                        cv2.imwrite(out_path + car.name[:-4] +  maskk.name[:-4] + scrach.name[:-4] + str(i)+str(j)+".jpg", cv2.resize(image[i*240:i*240+240,j*240:j*240+240], (160, 160)))
                        cv2.imwrite(mask_out_path +car.name[:-4]+ maskk.name[:-4] + scrach.name[:-4]+str(i)+str(j)+".jpg", cv2.resize(np.array(new_mask[i*240:i*240+240,j*240:j*240+240], dtype='uint8'), (160, 160)))

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


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="img/saved_images/", device=DEVICE
        )
import cv2
def bod():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint.pth.tar",map_location=torch.device('cpu')), model)
    val_transforms = A.Compose(
        [ A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ), ToTensorV2(),
        ],
    )

    val_ds = CarvanaDataset(
        image_dir="imgs\scratch_masks/test",
        mask_dir="imgs\scratch_masks/test_false",
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    save_predictions_as_imgs(
        val_loader, model, folder="imgs\scratch_masks/test_results", device=DEVICE
    )

def remove_files(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


if __name__ == "__main__":

    # bod()
    # segment(args.dir_path, args.output_dir)
    # augament(args.output_dir)
    # remove_files(VAL_MASK_DIR)
    # remove_files(VAL_IMG_DIR)
    # remove_files(TRAIN_IMG_DIR)
    # remove_files(TRAIN_MASK_DIR)
    # add_scratches()
    # #
    # #
    # for j,i in enumerate(os.listdir(TRAIN_IMG_DIR)):
    #     shutil.move(TRAIN_IMG_DIR+i,VAL_IMG_DIR+i)
    #     shutil.move(TRAIN_MASK_DIR+i,VAL_MASK_DIR+i)
    #     if j>1000:
    #         break
    # to255("img/bimaps")
    # to255("img/bi_val")
    # whiten()


    main()