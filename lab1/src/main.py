import cv2
import numpy as np
import matplotlib.pyplot as plt


def median_filter(img, k):
    w, h, c = img.shape
    size = k // 2

    # 0 padding process
    _img = np.zeros((w + 2 * size, h + 2 * size, c), dtype=float)
    _img[size:size + w, size:size + h] = img.copy().astype(float)
    dst = _img.copy()

    # Filtering process
    for x in range(w):
        for y in range(h):
            for z in range(c):
                dst[x + size, y + size] = np.median(_img[x:x + k, y:y + k, z])

    dst = dst[size:size + w, size:size + h].astype(np.uint8)
    return dst


if __name__ == "__main__":
    # Image reading
    image = cv2.imread('img/image.png')

    # Median filter
    filter_size = 2
    filtered_img = median_filter(image, filter_size)

    # Save image
    cv2.imwrite(f'result/result{filter_size}.png', filtered_img)
    # Image display
    plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
    plt.show()
