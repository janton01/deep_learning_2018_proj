single pic: 1024 * 1024

every *.npy file contains a single 2D numpy array 'img'. 
if there's N images, then the size of img is (1024*N, 1024). 

Data saved in this form saves much much more memory. 
Reshaping this array to 3D may be more handy for shuffle. 