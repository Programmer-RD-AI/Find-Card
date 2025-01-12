import os
import shutil
from time import sleep

import cv2
from tqdm import tqdm

shutil.copy("./img.png", "./Imgs/img.png")
shutil.copy("./img0.png", "./Imgs/img0.png")
shutil.copy("./img1.png", "./Imgs/img1.png")
shutil.copy("./img2.png", "./Imgs/img2.png")
data = {}
idx = 0
for file in tqdm(os.listdir("./Imgs/")):
    img = cv2.imread(f"./Imgs/{file}")
    hsv = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    )  # Convert the Image http://pythonwife.com/wp-content/uploads/image-38.png
    cv2.imwrite("hsv.png", hsv)  # Saving the hsv
    sat = hsv[:, :, 1]  # Convert the hsv Image to GrayScale
    cv2.imwrite("sat.png", sat)  # Saving the Sat
    thresh = cv2.threshold(sat, 20, 255, cv2.THRESH_BINARY)[
        1
    ]  # The Outline of the Sat which is the GrayScale Img
    cv2.imwrite("thresh.png", thresh)  # Saving the thresh
    thresh = 255 - thresh  # Converts Black to White and White to Black
    cv2.imwrite("thresh-2.png", thresh)  # Saving the thresh
    kernal_size_1 = 67
    kernal_size_2 = 350
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernal_size_1, kernal_size_2)
    )  # kernel which decides the nature of operation - OpenCV Documentation | What you can see visually is that the images lines get more bigger
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_DILATE, kernel
    )  # Helps to remove noise https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    cv2.imwrite(
        "black_and_white.png", thresh
    )  # Saving the black and white image which is the edge of the NCIS Card
    im = cv2.imread("black_and_white.png")  # Reading The Edge Image
    gray = cv2.cvtColor(
        im, cv2.COLOR_BGR2GRAY
    )  # Conver the black and white image to a image like https://i.stack.imgur.com/UPOZC.png
    cv2.imwrite("gray.png", gray)  # Saving the gray image
    contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
        -2:
    ]  # Finding the boxes

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Convert cnt to x,y,w,h
        if (
            w > 250 and h > 250
        ):  # Checking if the h and w of the image is higher than 175 so this will only get the card
            idx += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0))
            img = cv2.imread(f"./Imgs/{file}")  # get the original image
            crop_img = img[y : y + h, x : x + w]  # Cropping
            if f"./Imgs/{file}" in list(data):
                data[f"{file}"]["X"].append(x)
                data[f"{file}"]["Y"].append(y)
                data[f"{file}"]["W"].append(w)
                data[f"{file}"]["H"].append(h)
            else:
                data[f"{file}"] = {"X": [], "Y": [], "W": [], "H": []}
                data[f"{file}"]["X"].append(x)
                data[f"{file}"]["Y"].append(y)
                data[f"{file}"]["W"].append(w)
                data[f"{file}"]["H"].append(h)
    sleep(5)
for key, val in zip(list(data.keys()), list(data.values())):
    X = max(val["X"])
    Y = max(val["Y"])
    W = int(max(val["W"]) - int(kernal_size_2 / 2))
    H = int(max(val["H"]) - int(kernal_size_2 / 2))
    print(X, Y, W, H)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0))
    img = cv2.imread(f"./Imgs/{key}")  # get the original image
    cv2.imwrite(
        f"./Preds/{file}.jpeg",
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (200, 0, 0)),
    )  # Saving the corped image
    crop_img = img[Y : Y + H, X : X + W]  # Cropping
    cv2.imwrite(f"./Imgs/{key}", crop_img)  # Saving the corped image
