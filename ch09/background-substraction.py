# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    img_src = cv2.imread('../images/Label.jpg', cv2.IMREAD_GRAYSCALE)
    img_bkg = cv2.imread('../images/Mask.jpg', cv2.IMREAD_GRAYSCALE)

    # Background Substraction
    img_diff = cv2.absdiff(img_src, img_bkg)
    ret, img_bin = cv2.threshold(img_diff, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)

    # 膨張・収縮処理で小さな孔や連結部分を除去
    img_dilation = cv2.dilate(img_bin, kernel, iterations=4)  # 膨張処理
    img_msk = cv2.erode(img_dilation, kernel, iterations=4)  # 収縮処理

    img_dst = cv2.bitwise_and(img_src, img_msk)  # マスク画像で切り出す

    images = [img_src, img_bkg, img_dst]
    titles = ['original', 'background', 'background substraction']

    for i in range(3):
        plt.subplot(1, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.savefig('../output/cv2-background-sub.jpg')
    plt.show()
