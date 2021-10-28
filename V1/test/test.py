import cv2

im = cv2.imread("white_black_text_threshold.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[
    -2:
]
idx = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 100 and h > 100:
        idx += 1
        roi = im[y : y + h, x : x + w]
        cv2.imwrite(str(idx) + ".jpg", roi)
        cv2.rectangle(im, (x, y), (x + w, y + h), (200, 0, 0), 2)
cv2.imshow("img", im)
cv2.waitKey(0)
