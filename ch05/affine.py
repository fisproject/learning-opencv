# -*- coding: utf-8 -*-

import cv2
import numpy as np

if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_UNCHANGED)

    height, width = img_src.shape[:2]
    size = tuple(np.array([width, height]))

    rad = np.pi/6 # 回転角度
    move_x = width*0.1 # x方向平行移動
    move_y = height*-0.3 # y方向平行移動

    afn_mat = np.float32(
        [[np.cos(rad), -1*np.sin(rad), move_x], [np.sin(rad), np.cos(rad), move_y]]
    )

    img_dst = cv2.warpAffine(img_src, afn_mat, size, flags=cv2.INTER_LINEAR)

    cv2.imshow('Affine Transformation', img_dst)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite('../output/cv2-affine.jpg', img_dst)
    cv2.destroyAllWindows()
