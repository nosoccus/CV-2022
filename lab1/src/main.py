import cv2
import numpy as np
import matplotlib.pyplot as plt

from noise import add_noise


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

    noise_img = add_noise(image)
    cv2.imwrite('img/noised.png', noise_img)
    plt.imshow(cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB))
    plt.show()

    filter_sizes = [1, 2, 5, 10, 20, 50]
    for filter_size in filter_sizes:
        filtered_img = median_filter(noise_img, filter_size)

        cv2.imwrite(f'results/result{filter_size}.png', filtered_img)
        # plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
        # plt.show()
