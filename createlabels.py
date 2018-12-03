import numpy as np

f=open('labels.txt', 'rt')
lines = f.readlines()
labels=[]
for line in lines:
    i = line.find('\t')
    labels.append(line[i+1:i+2])

pic_idx=np.load('picture_index.npy')
report_idx = np.load('report_idx.npy')
output_label = np.zeros_like(pic_idx)
for i in range(pic_idx.shape[0]):
    k = np.where(report_idx==pic_idx[i])
    output_label[i]=int(labels[k[0][0]])
tmp_label = output_label
for k in range(6):
    tmp_label = np.vstack((tmp_label,output_label))
output_label = np.transpose(tmp_label).reshape(-1,)
np.save('labels.npy', output_label)