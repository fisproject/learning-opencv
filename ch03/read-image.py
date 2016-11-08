# -*- coding: utf-8 -*-

import cv2

if __name__ == "__main__":
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_UNCHANGED)

    height, width, channels = img_src.shape[:3]

    print('width: %s, height: %s, channels: %s, dtype: %s' %
        (str(width), str(height), str(channels), str(img_src.dtype)))
    # width: 400, height: 400, channels: 3, dtype: uint8
