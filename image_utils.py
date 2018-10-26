import numpy as np
import cv2
from skimage.transform import resize

def im_max_pool(im, kernel=(2,2), anti_aliasing=True):
    kernel = np.array(kernel, dtype=np.int32)
    assert kernel.shape == (2,), 'kernel must be of shape (2,)'

    if anti_aliasing:
        anti_aliasing_sigma = np.round(np.maximum(np.zeros_like(kernel), (kernel-1.) / 2.)).astype(np.int32)
        #anti_aliasing_sigma = np.round(np.maximum(np.zeros_like(kernel), (kernel-1.))).astype(np.int32)
        # sigma needs to be integer = 1 mod 2
        anti_aliasing_sigma += (1 - anti_aliasing_sigma % 2)
        im = cv2.GaussianBlur(im, tuple(anti_aliasing_sigma), 0)

    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    kernel = np.concatenate((kernel, [1]))
    pad_dims = [
        [(im.shape[0]%kernel[0]+1) // 2, (im.shape[0]%kernel[0]) // 2],
        [(im.shape[1]%kernel[1]+1) // 2, (im.shape[1]%kernel[1]) // 2],
        [0,0],
    ]
    im = np.pad(im, pad_dims, mode='reflect')
    new_im_shape = [
        im.shape[0] // kernel[0],
        im.shape[1] // kernel[1],
        im.shape[2]
    ]
    new_im = np.zeros(new_im_shape, dtype=im.dtype)
    for y in range(new_im.shape[0]):
        for x in range(new_im.shape[1]):
            reduce_region = im[
                max(kernel[0]*(y-0), 0):min(kernel[0]*(y+1), im.shape[0]),
                max(kernel[1]*(x-0), 0):min(kernel[1]*(x+1), im.shape[1]),
                :]
            mean_region = im[
                max(kernel[0]*(y-1), 0):min(kernel[0]*(y+2), im.shape[0]),
                max(kernel[1]*(x-1), 0):min(kernel[1]*(x+2), im.shape[1]),
                :]
            pxwise_mean = np.mean(np.mean(mean_region, axis=0), axis=0)
            diff = reduce_region - np.expand_dims(np.expand_dims(pxwise_mean, axis=0), axis=0)
            score = np.mean(np.square(diff), axis=2)
            outlier_nd = np.unravel_index(np.argmax(score, axis=None), score.shape)
            new_im[y, x] = reduce_region[outlier_nd]
            new_im[y, x] = np.mean(np.mean(reduce_region, axis=0), axis=0)
    return new_im

if __name__ == '__main__':
    #im = cv2.imread('/home/alvin/Pictures/tmp.jpg')
    #im = cv2.imread('/home/alvin/Downloads/atari.jpg')
    #im = cv2.imread('/home/alvin/Downloads/atari.png')
    im = cv2.imread('/home/alvin/Downloads/spaceinvaders.png')
    cv2.imshow('orig_im', im)
    # max_im = im_max_pool(im, kernel=(4,4), anti_aliasing=True)
    # cv2.imshow('max_im', max_im)
    # mean_im = resize(im, (im.shape[0]//4, im.shape[1]//4), anti_aliasing=True)
    # cv2.imshow('mean_im', mean_im)
    mean_im = cv2.resize(im, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imshow('cv2_area_im', mean_im)
    # mean_im = cv2.resize(im, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('cv2_cubic_im', mean_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
