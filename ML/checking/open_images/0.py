info = data_other.iloc[data_idx]  # getting the info of the index
img = cv2.imread(f'./Img/{info["Path"]}')  # reading the img
height, width = cv2.imread("./Img/" + info["Path"]).shape[
    :2
]  # getting the height and width of the image
xmin, ymin, xmax, ymax = (
    info["XMin"],
    info["YMin"],
    info["XMax"],
    info["YMax"],
)  # getting the xmin,ymin, xmax, ymax
xmin = round(xmin * width)  # converting it to real xmin
xmax = round(xmax * width)  # converting it to real xmax
ymin = round(ymin * height)  # converting it to real ymin
ymax = round(ymax * height)  # converting it to real ymax
# The above is needed becuase open images gives their datasets xmin,ymin,xmax,ymax in a different way
x = xmin
y = ymin
w = xmax - xmin
h = ymax - ymin
x, y, w, h = round(x), round(y), round(w), round(h)
roi = img[y : y + h, x : x + w]  # crop the image
cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 10)  # draw box around the bbox
