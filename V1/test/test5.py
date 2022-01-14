import os
import cv2
import numpy as np

data = {}
idx = 0
for file in os.listdir("./Imgs/"):
    file = "0.jpeg"
    img = cv2.imread(f"./Imgs/{file}")
    # img = cv2.resize(img, (1400, 1200))
    hsv = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    )  # Convert the Image http://pythonwife.com/wp-content/uploads/image-38.png
    cv2.imwrite("hsv.png", hsv)  # Saving the hsv
    sat = hsv[:, :, 1]  # Convert the hsv Image to GrayScale
    cv2.imwrite("sat.png", sat)  # Saving the Sat
    thresh = cv2.threshold(sat, 10, 255, cv2.THRESH_BINARY)[
        1]  # The Outline of the Sat which is the GrayScale Img
    cv2.imwrite("thresh.png", thresh)  # Saving the thresh
    thresh = 255 - thresh  # Converts Black to White and White to Black
    cv2.imwrite("thresh-2.png", thresh)  # Saving the thresh
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)
    )  # kernel which decides the nature of operation - OpenCV Documentation | What you can see visually is that the images lines get more bigger
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_DILATE, kernel
    )  # Helps to remove noise https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    cv2.imwrite(
        "black_and_white.png", thresh
    )  # Saving the black and white image which is the edge of the NCIS Card
    im = cv2.imread("black_and_white.png")  # Reading The Edge Image
    # im = cv2.resize(im, (1400, 1200))
    gray = cv2.cvtColor(
        im, cv2.COLOR_BGR2GRAY
    )  # Conver the black and white image to a image like https://i.stack.imgur.com/UPOZC.png
    cv2.imwrite("gray.png", gray)  # Saving the gray image
    contours, _ = cv2.findContours(
        gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]  # Finding the boxes

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # Convert cnt to x,y,w,h
        if (
                w > 125 and h > 125
        ):  # Checking if the h and w of the image is higher than 175 so this will only get the card
            idx += 1
            print(x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0))
            img = cv2.imread(f"./Imgs/{file}")  # get the original image
            # img = cv2.resize(img, (1400, 1200))
            # y = cnt[1]  # Cropping y
            # x = cnt[0]  # Cropping x
            # h = cnt[1]  # Cropping h
            # w = cnt[0]  # Cropping w
            crop_img = img[y:y + h, x:x + w]  # Cropping
            cv2.imwrite(
                f"./Preds/{file}-{idx}.jpeg",
                cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0)),
            )
            print(x + w, y + h)

            if f"./Imgs/{file}" in list(data.keys()):
                print(data)
                data[f"./Imgs/{file}"]["X"].append(x)
                data[f"./Imgs/{file}"]["Y"].append(y)
                data[f"./Imgs/{file}"]["W"].append(w)
                data[f"./Imgs/{file}"]["H"].append(h)
            else:
                print(data)
                data[f"./Imgs/{file}"] = {"X": [], "Y": [], "W": [], "H": []}
                data[f"./Imgs/{file}"]["X"].append(x)
                data[f"./Imgs/{file}"]["Y"].append(y)
                data[f"./Imgs/{file}"]["W"].append(w)
                data[f"./Imgs/{file}"]["H"].append(h)
for key, val in zip(list(data.keys()), list(data.values())):
    X = max(val["X"])
    Y = max(val["Y"])
    W = max(val["W"])
    H = max(val["H"])
    print(X, Y, W, H)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0))
    img = cv2.imread(f"./Imgs/{file}")  # get the original image
    # img = cv2.resize(img, (1400, 1200))
    crop_img = img[Y:Y + H, X:X + W]  # Cropping
    cv2.imwrite(f"./Preds/{file}-{idx}-crop.jpeg",
                crop_img)  # Saving the corped image
print(data)
print(list(data.keys()))
