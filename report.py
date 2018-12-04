import numpy as np
pic_idx=np.load('picture_index.npy')
report_idx = np.load('report_idx.npy')
report = np.load('converted_report.npy')

output_label = np.empty_like(pic_idx, dtype=np.object)

for i in range(pic_idx.shape[0]):
    k = np.where(report_idx==pic_idx[i])
    output_label[i]=report[k[0][0]]
tmp_label = output_label
for k in range(6):
    tmp_label = np.vstack((tmp_label,output_label))
output_label = np.transpose(tmp_label).reshape(-1,)
np.save('report.npy',output_label)
