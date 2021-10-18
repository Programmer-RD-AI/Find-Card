import cv2
import numpy as np

# read input
img = cv2.imread('./Imgs/WhatsApp Image 2021-09-28 at 09.45.58 (1).jpeg')

# convert to hsv and extract saturation
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
sat = hsv[:,:,1]

# threshold and invert
thresh = cv2.threshold(sat, 10, 255, cv2.THRESH_BINARY)[1]
thresh = 255 - thresh

# apply morphology dilate
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# do inpainting
result1 = cv2.inpaint(img,thresh,11,cv2.INPAINT_TELEA)
result2 = cv2.inpaint(img,thresh,11,cv2.INPAINT_NS)

# save results
cv2.imwrite('white_black_text_threshold.png', thresh)
cv2.imwrite('white_black_text_inpainted1.png', result1)
cv2.imwrite('white_black_text_inpainted2.png', result1)

# show results
cv2.imshow('thresh',thresh)
cv2.imshow('result1',result1)
cv2.imshow('result2',result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
