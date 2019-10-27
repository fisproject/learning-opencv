# -*- coding: utf-8 -*-

import cv2
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)

    # to HSV
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)

    plt.imshow(img_hsv)
    plt.title('bgr2hsvs')
    plt.savefig('../output/cv2-bgr2hsv.jpg')
    plt.show()

    # to GRAY
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # matplotlib require RGB format
    cv2.imshow('COLOR_BGR2GRAY', img_gry)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite('../output/cv2-bgr2gray.jpg', img_gry)
    cv2.destroyAllWindows()
