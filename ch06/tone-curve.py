# -*- coding: utf-8 -*-

import cv2
import numpy as np

def histogram(src):
    img_hist = np.zeros([100, 256]).astype("uint8") # Init [100, 256]
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

def craete_table(shift):
    t = np.arange(256, dtype=np.uint8)
    for i in xrange(0, 255):
        j = i + shift
        if j < 0:
            t[i] = 0
        elif j > 255:
            t[i] = 255
        else:
            t[i] = j
    return t

def craete_enhancement_table(min, max):
    t = np.arange(256, dtype=np.uint8)
    for i in xrange(0, min):
        t[i] = 0
    for i in xrange(min, max):
        t[i] = 255 * (i - min) / (max - min)
    for i in xrange(max, 255):
        t[i] = 255
    return t

def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    # ネガポジ変換
    img_dst = 255 - img_gry
    show_image('NP', img_dst)
    cv2.imwrite('../output/cv2-np.jpg', img_dst)

    # ネガポジ変換後のヒストグラム
    img_hist = histogram(img_dst)
    show_image('Histogram of NP', img_hist)
    cv2.imwrite('../output/cv2-hist-of-np.jpg', img_hist)

    # 明度調整
    table = craete_table(shift=100)
    img_dst = cv2.LUT(img_gry, table)
    show_image('Value Adjustment', img_dst)
    cv2.imwrite('../output/cv2-value-adjustment.jpg', img_dst)

    # 明度調整後のヒストグラム
    img_hist = histogram(img_dst)
    show_image('Histogram of Value Adjustment', img_hist)
    cv2.imwrite('../output/cv2-hist-of-va.jpg', img_hist)

    # コントラスト低減
    cv2.normalize(img_gry, img_dst, alpha=100, beta=200, norm_type=cv2.NORM_MINMAX)
    show_image('Contrast Reducing', img_dst)
    cv2.imwrite('../output/cv2-contrast-reducing.jpg', img_dst)

    # コントラスト低減後のヒストグラム
    img_hist = histogram(img_dst)
    show_image('Histogram of Contrast Reducing', img_hist)
    cv2.imwrite('../output/cv2-hist-of-cr.jpg', img_hist)

    # コントラスト強調
    table = craete_enhancement_table(min=150, max=200)
    img_dst = cv2.LUT(img_gry, table)
    show_image('Contrast Enhancement', img_dst)
    cv2.imwrite('../output/cv2-contrast-enhancement.jpg', img_dst)

    # コントラスト強調後のヒストグラム
    img_hist = histogram(img_dst)
    show_image('Histogram of Contrast Enhancement', img_hist)
    cv2.imwrite('../output/cv2-hist-of-ce.jpg', img_hist)

    cv2.destroyAllWindows()
