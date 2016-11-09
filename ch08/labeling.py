# -*- coding: utf-8 -*-

import cv2
import copy
import random
import numpy as np

if __name__ == '__main__':
    img_src = cv2.imread('../images/Label.jpg', cv2.IMREAD_COLOR)
    height, width = img_src.shape[:2]
    img_dst = copy.copy(img_src)
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # ラベリング
    ret, th = cv2.threshold(img_gry, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    labels, img_obj = cv2.connectedComponents(th)
    print('labels: %s' % labels)

    colors = []
    for i in xrange(1, labels+1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    for y in xrange(0, height):
        for x in xrange(0, width):
            if img_obj[y, x] > 0:
                img_dst[y, x] = colors[img_obj[y, x]]
            else:
                img_dst[y, x] = [0, 0, 0]

    output = cv2.connectedComponentsWithStats(th, connectivity=4, ltype=cv2.CV_32S)
    labels, labels, stats, centroids = output[:4]
    print(centroids)

    cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Source", img_src)
    cv2.namedWindow("Connected Components", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Connected Components", img_dst)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imwrite('../output/cv2-labeling.jpg', img_dst)
    cv2.destroyAllWindows()
