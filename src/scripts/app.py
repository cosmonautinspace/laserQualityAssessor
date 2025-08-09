import scipy.io
mat = scipy.io.loadmat('data/laser.mat')
print(mat['X'].shape)
