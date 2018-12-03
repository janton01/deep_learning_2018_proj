import numpy as np
import os
import cv2
import preprocessing as pp
import stretch
import matplotlib.pyplot as plt

f=open('labels.txt', 'rt')
lines = f.readlines()
out = []
t=0
report_idx = np.zeros(len(lines),dtype=int)
diction = []
for line in lines:
    line = line.replace('"','')
    line = line.replace('(', ' ( ')
    line = line.replace(')', ' ) ')
    line = line.replace('.', ' . ')
    line = line.replace(',', ' , ')
    i=line.find('\t')
    report_idx[t] = int(line[:i])
    t=t+1
    if 10> int(line[:i]) / 10 >= 1:
        out.append(line[5:-1].upper())
        k=line[5:-1]
        k = k.upper().split(' ')
        for j in range(len(k)):
            diction.append(k[j])
    elif 10>int(line[:i])/100>=1:
        out.append(line[6:-1].upper())
        k=line[6:-1]
        k = k.upper().split(' ')
        for j in range(len(k)):
            diction.append(k[j])
    elif 10 > int(line[:i]) / 1000 >= 1:
        out.append(line[7:-1].upper())
        k=line[7:-1]
        k = k.upper().split(' ')
        for j in range(len(k)):
            diction.append(k[j])
    else:
        out.append(line[4:-1].upper())
        k=line[4:-1]
        k = k.upper().split(' ')
        for j in range(len(k)):
            diction.append(k[j])
out = np.array(out)
diction = np.array(diction)
k = np.unique(diction)
idx = np.array(range(k.shape[0]))
diction = dict(zip(k,idx))
out_tmp = np.empty(out.shape[0],dtype=np.object)
for i in range(out.shape[0]):
    input = out[i].split(' ')
    tmp = np.zeros(len(input),dtype=int)
    for k in range(len(input)):
        tmp[k] = diction[input[k]]
    out_tmp[i] = tmp
np.save('converted_report.npy',out_tmp)
list_pic = os.listdir('pngdata')
pic_index = np.zeros(len(list_pic),dtype=int)
Big_pic = np.zeros((256,256,len(list_pic)*7))
j=0
MAX = 255
MIN = 0
for path in list_pic:
    pic_index[j]=int(path[3:path.find('_')])
    Img =cv2.imread('pngdata/'+path,0)
    rotated_img = pp.create_two_random_rotations(Img, 2, 5, 15)
    st_img1, st_img2 = stretch.stretch_img(Img)
    Imax = np.max(Img)
    Imin = np.min(Img)
    Big_pic[:,:,(3*j)] = cv2.resize(Img, (256, 256))
    contrast = np.random.randint(50, 100)
    Big_pic[:, :, (3*j+1)] = cv2.resize(((Img - Imin) / (Imax - Imin) * (MAX - MIN) + MIN), (256, 256))
    Big_pic[:, :, (3 * j + 2)] = cv2.resize((Img+contrast), (256, 256))
    Big_pic[:, :, (3 * j + 3)] = cv2.resize(rotated_img[0], (256, 256))
    Big_pic[:, :, (3 * j + 4)] = cv2.resize(rotated_img[1], (256, 256))
    Big_pic[:, :, (3 * j + 5)] = cv2.resize(st_img1, (256, 256))
    Big_pic[:, :, (3 * j + 6)] = cv2.resize(st_img2, (256, 256))
    j = j+1
np.save('picture_data.npy', Big_pic)
np.save('picture_index.npy', pic_index)
np.save('report_idx.npy', report_idx)
a=0