import cv2
import numpy as np

# Read
img = cv2.imread("./Imgs/WhatsApp Image 2021-09-28 at 09.45.58 (1).jpeg")

# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# mask of green (36,25,25) ~ (86, 255,255)
# mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
mask = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

# slice the green
imask = mask > 0
green = np.zeros_like(img, np.uint8)
green[imask] = img[imask]

# save
cv2.imwrite("green.png", green)
