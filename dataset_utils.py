import numpy as np
import cv2
import h5py

data = h5py.File('data/obj_balls.h5', 'r')
#import ipdb; ipdb.set_trace()
print('loading im')
# ims = data['training']['groups'][:,0,:,:,0].astype(np.uint8)*63
ims = data['training']['features'][:,0,:,:,0].astype(np.uint8)*255
print('done')
for t in range(50):
    cv2.imshow('im', ims[t])
    cv2.waitKey(100)
cv2.destroyAllWindows()
