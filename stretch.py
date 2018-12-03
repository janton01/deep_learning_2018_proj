import numpy as np
import cv2 as cv

# read images


def stretch_img(img):
    y = img.shape[0]
    x = img.shape[1]
    y_squeeze = int(np.round(y * np.random.randint(85, 90) / 100))
    x_squeeze = int(np.round(x * np.random.randint(85, 90) / 100))
    img_y = cv.resize(img, (x, y_squeeze))
    img_x = cv.resize(img, (x_squeeze, y))
    return img_x, img_y
