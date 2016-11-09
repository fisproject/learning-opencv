# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # モフォロジー処理
    kernel = np.ones((5,5), np.uint8)
    img_elosion = cv2.erode(img_gry, kernel, iterations=1) # 収縮処理
    img_dilation = cv2.dilate(img_gry, kernel, iterations=1) # 膨張処理
    img_close = cv2.morphologyEx(img_gry, cv2.MORPH_CLOSE, kernel) # クロージング
    img_open = cv2.morphologyEx(img_gry, cv2.MORPH_OPEN, kernel) # オープニング

    images = [img_src, img_gry, img_elosion, img_dilation, img_close, img_open]
    titles = ['original', 'gray', 'elosion', 'dilation', 'closing', 'opening']

    for i in xrange(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.savefig('../output/cv2-morphology.jpg')
    plt.show()
