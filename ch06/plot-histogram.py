# -*- coding: utf-8 -*-

import cv2
import numpy as np

def histogram(src):
    img_hist = np.zeros([100, 256]).astype("uint8") # # Init [100, 256]
    rows, cols = img_hist.shape

    # 度数分布
    hdims = [256]
    hranges = [0, 256]
    hist = cv2.calcHist([src], [0], None, hdims, hranges)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)

    for i in range(0, 255):
        v = hist[i]
        cv2.line(img_hist, (i, rows), (i, rows - rows * (v / max_val)), (255, 255, 255))

    return img_hist

def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # ヒストグラム
    img_hist = histogram(img_gry)
    show_image('Histogram', img_hist)
    cv2.imwrite('../output/cv2-hist-of-org.jpg', img_hist)

    # ヒストグラム均一化
    img_dst = cv2.equalizeHist(img_gry)
    show_image('Equalize Histogram', img_dst)
    cv2.imwrite('../output/cv2-equalize-hist.jpg', img_dst)

    cv2.destroyAllWindows()
