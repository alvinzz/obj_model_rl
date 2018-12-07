import pickle; import numpy as np; import cv2
im = pickle.load(open('data/ae_relu_kl_3_07-12-2018_06-03/5/sampled_4.pkl', 'rb'))
new_im = np.zeros((64, 64, 3), dtype=np.uint8)
num_active_layers = 0
for i in range(8):
    if np.max(im[:,:,i]) > 0:
        num_active_layers += 1
        color = [int((i%8)>=4),int((i%4)>=2),int((i%2)>=1)]
        new_im += np.round(im[:,:,i]).astype(np.uint8).reshape(64, 64, 1) \
            *255//2 \
            *np.array(color).astype(np.uint8).reshape(1, 1, 3)
print(num_active_layers)
cv2.imshow('im', new_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
