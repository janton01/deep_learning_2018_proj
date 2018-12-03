import cv2
import random

"""
    Pass in an image and creates num_imgs rotated image
    which are each rotated by a random amount between
    +min rotation and + max_rotation or between -min rotation
    and -max rotation.
"""

'''
    example of how the function to create rotated imgs works
'''

def main():
    img = cv2.imread("pngdata/CXR1_1_IM-0001-4001.png", 0)
    num_imgs = 10
    min_rot = 5
    max_rot = 20
    out = create_two_random_rotations(img, num_imgs, min_rot, max_rot)
    for i in out:
        cv2.imshow('img',i)
        cv2.waitKey(0)

def create_two_random_rotations(img, num_imgs, min_rot_deg, max_rot_deg):
    out_imgs = []
    for i in range(num_imgs):
        rows, cols = img.shape
        rot_deg = random.randint(min_rot, max_rot)
        sign = -1 if random.randint(0, 1) == 1 else 1

        rot_mat = cv2.getRotationMatrix2D((cols / 2,rows / 2), sign * rot_deg,1)
        dst = cv2.warpAffine(img, rot_mat, (cols, rows))
        out_imgs.append(dst)
    return out_imgs


if __name__ == '__main__':
    main()
