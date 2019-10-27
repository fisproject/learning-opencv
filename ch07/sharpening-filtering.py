# -*- coding: utf-8 -*-

import cv2
import numpy as np


def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)

    # 鮮鋭化
    k = 1.0
    op = np.array([[-k, -k, -k], [-k, 1+8*k, -k], [-k, -k, -k]])
    img_tmp = cv2.filter2D(img_src, ddepth=-1, kernel=op)
    img_dst = cv2.convertScaleAbs(img_tmp)
    show_image('Sharpening', img_dst)
    cv2.imwrite('../output/cv2-sharpening.jpg', img_dst)

    cv2.destroyAllWindows()
