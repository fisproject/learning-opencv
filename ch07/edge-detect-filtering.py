# -*- coding: utf-8 -*-

import cv2


def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    img_src = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_COLOR)

    # Sobelオペレータ
    img_tmp = cv2.Sobel(img_src, ddepth=cv2.CV_32F, dx=1, dy=0)
    img_dst = cv2.convertScaleAbs(img_tmp)
    show_image('Sobel', img_dst)
    cv2.imwrite('../output/cv2-sobel.jpg', img_dst)

    cv2.destroyAllWindows()
