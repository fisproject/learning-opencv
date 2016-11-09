# -*- coding: utf-8 -*-

import cv2
import numpy as np

def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_GRAYSCALE)
    img_msk = cv2.imread('../images/Mask.jpg', cv2.IMREAD_GRAYSCALE)

    # マスク処理
    img_dst = cv2.bitwise_and(img_src, img_src, mask=img_msk)
    show_image('Masking', img_dst)
    cv2.imwrite('../output/cv2-masking.jpg', img_dst)
    cv2.destroyAllWindows()
