# -*- coding: utf-8 -*-

import cv2


def show_image(title, data):
    cv2.imshow(title, data)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    img_src1 = cv2.imread('../images/Lenna.jpg', cv2.IMREAD_GRAYSCALE)
    img_src2 = cv2.imread('../images/Mask.jpg', cv2.IMREAD_GRAYSCALE)

    # Alpha Blending
    img_dst = cv2.addWeighted(
                src1=img_src1,
                alpha=0.5,
                src2=img_src2,
                beta=0.5,
                gamma=0.0
              )
    show_image('Alpha Blending', img_dst)
    cv2.imwrite('../output/cv2-alpha-blending.jpg', img_dst)
    cv2.destroyAllWindows()
