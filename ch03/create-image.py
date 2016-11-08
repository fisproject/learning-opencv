# -*- coding: utf-8 -*-

import cv2
import numpy as np

if __name__ == "__main__":
    cols = 480
    rows = 640
    img_blk = np.zeros((rows, cols, 3), np.uint8) # numpy.ndarray

    height, width, channels = img_blk.shape[:3]
    print('width: %s, height: %s, channels: %s, dtype: %s' %
        (str(width), str(height), str(channels), str(img_blk.dtype)))

    cv2.imshow('blank_image', img_blk)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite('../output/cv2-black.png', img_blk)
    cv2.destroyAllWindows()
