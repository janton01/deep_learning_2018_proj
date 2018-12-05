import numpy as np
import cv2 as cv

# read images
# cv.resize(img, (x, y_squeeze))
# img_x = cv.resize(img, (x_squeeze, y))


def changing_img(img):
    # img = cv.imread(path, 0)
    y = img.shape[0]
    x = img.shape[1]
    y_squeeze = int(np.round(y * np.random.randint(6, 10) / 100))
    x_squeeze = int(np.round(x * np.random.randint(6, 10) / 100))
    img_y = np.zeros((y + 2 * y_squeeze, x), dtype=np.uint8)
    img_x = np.zeros((y, x + 2 * x_squeeze), dtype=np.uint8)
    img_padded = np.zeros((y + 2 * y_squeeze, x + 2 * x_squeeze), dtype=np.uint8)
    img_y[y_squeeze:y_squeeze + y, :] = img
    img_x[:, x_squeeze:x_squeeze + x] = img
    img_padded[y_squeeze:y_squeeze + y, x_squeeze:x_squeeze + x] = img
    img_cropped = img[y_squeeze:y - y_squeeze, x_squeeze:x-x_squeeze].copy()
    return img_x, img_y, img_padded, img_cropped


# path = 'pngdata/CXR7_IM-2263-1001.png'
# [img_x, img_y, img_padded, img_cropped] = changing_img(path)
# img = img_cropped
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# # a = 0