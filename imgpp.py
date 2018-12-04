import numpy as np
import imgpreprocess
data = np.load('data/picture_data.npy')
data = data.transpose((2,0,1))
data_output = np.zeros_like(data)
for i in range(data.shape[0]):
    data_output[i,:,:] = imgpreprocess.cifar_10_preprocess(data[i,:,:], image_size=256)
    if i % 1000 == 0
    print('Working')
data_output = data_output.transpose((1,2,0))
np.save('picture_data_processed.npy', data_output)