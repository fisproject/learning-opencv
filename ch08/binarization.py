# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # 二値化
    thresholds = [50, 100, 150, 200, 250]
    ret, img_dst1 = cv2.threshold(img_gry, thresholds[0], 255, cv2.THRESH_BINARY)
    ret, img_dst2 = cv2.threshold(img_gry, thresholds[1], 255, cv2.THRESH_BINARY)
    ret, img_dst3 = cv2.threshold(img_gry, thresholds[2], 255, cv2.THRESH_BINARY)
    ret, img_dst4 = cv2.threshold(img_gry, thresholds[3], 255, cv2.THRESH_BINARY)
    ret, img_dst5 = cv2.threshold(img_gry, thresholds[4], 255, cv2.THRESH_BINARY)

    titles = ['original','threshold=50','threshold=100',
            'threshold=150','threshold=200','threshold=250']

    images = [img_gry, img_dst1, img_dst2, img_dst3, img_dst4, img_dst5]

    for i in xrange(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

    plt.savefig('../output/cv2-binarization.jpg')
    plt.show()
