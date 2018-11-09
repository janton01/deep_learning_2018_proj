import scipy.io as sio
import numpy as np


name_abnormal = np.array(['abnormal_batch_1_1.mat',
                          'abnormal_batch_1_2.mat',
                          'abnormal_batch_1_3.mat'])
# name_normal = np.array(['normal_batch_1_1.mat',
#                         'normal_batch_1_2.mat',
#                         'normal_batch_2_1.mat',
#                         'normal_batch_2_2.mat',
#                         'normal_batch_3_1.mat',
#                         'normal_batch_3_2.mat',
#                         'normal_batch_4.mat',
#                         'normal_batch_5.mat',
#                         'normal_batch_6.mat',
#                         'normal_batch_7.mat',
#                         'normal_batch_8.mat',
#                         'normal_batch_9.mat',
#                         'normal_batch_10_1.mat',
#                         'normal_batch_10_2.mat',
#                         'normal_batch_11.mat',
#                         'normal_batch_12.mat'])
# abnormal data_set
what_ever_name = sio.loadmat(name_abnormal[0])
aha = what_ever_name['img_3d']
abnormal_data = np.squeeze(aha, axis=1)
for i in range(name_abnormal.shape[0] - 1):
    what_ever_name = sio.loadmat(name_abnormal[i + 1])
    aha = what_ever_name['img_3d']
    abnormal_data = np.hstack([abnormal_data, np.squeeze(aha, axis=1)])

import pdb
pdb.set_trace()

"""
# normal data_set
what_ever_name = sio.loadmat(name_normal[0])
aha = what_ever_name['img_3d']
normal_data = np.squeeze(aha, axis=1)
for i in range(name_normal.shape[0] - 1):
    what_ever_name = sio.loadmat(name_normal[i + 1])
    aha = what_ever_name['img_3d']
    normal_data = np.hstack([normal_data, np.squeeze(aha, axis=1)])
"""
