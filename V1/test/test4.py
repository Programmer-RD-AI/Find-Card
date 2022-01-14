import cv2

img = cv2.imread("./Imgs/0.jpeg")  # get the original image
img = cv2.resize(img, (1080, 1080))
y = 100  # Cropping y
x = 100  # Cropping x
h = 900  # Cropping h
w = 900  # Cropping w
crop_img = img[y:y + h, x:x + w]  # Cropping
cv2.imwrite("./Preds/cropped.jpeg", crop_img)  # Saving the corped imag
