# -*- coding: utf-8 -*-

import cv2


def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)

    # 平均化オペレータ
    img_dst = cv2.blur(img_src, ksize=(5, 5))
    show_image('Normalized box filter', img_dst)
    cv2.imwrite('../output/cv2-normalized.jpg', img_dst)

    # Gaussianオペレータ
    img_dst = cv2.GaussianBlur(img_src, ksize=(5, 5), sigmaX=1)
    show_image('Gaussian', img_dst)
    cv2.imwrite('../output/cv2-gaussian.jpg', img_dst)

    # Bilateralオペレータ
    img_dst = cv2.bilateralFilter(img_src, d=5, sigmaColor=50, sigmaSpace=100)
    show_image('Bilateral', img_dst)
    cv2.imwrite('../output/cv2-bilateral.jpg', img_dst)

    # medianオペレータ
    img_dst = cv2.medianBlur(img_src, ksize=5)
    show_image('Median', img_dst)
    cv2.imwrite('../output/cv2-median.jpg', img_dst)

    cv2.destroyAllWindows()
