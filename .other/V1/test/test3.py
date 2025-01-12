import cv2

img = cv2.imread("img.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (10, 100, 20), (25, 255, 255))
cv2.imshow("orange", mask)
