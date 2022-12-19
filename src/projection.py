import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

MAX_PLOT_HEIGHT = 100


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def remove_dots(img, threshold=11):
    # Remove connected components that are too small (the dots)
    no_dots_img = img.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(no_dots_img, connectivity=8)
    remaining_component_labels = set()
    for label in range(1, num_labels):
        _, _, _, _, size = stats[label]
        if size > threshold:
            remaining_component_labels.add(label)
    for label in range(1, num_labels):
        if label not in remaining_component_labels:
            no_dots_img[labels == label] = 0

    return no_dots_img


def baseline(img):
    no_dots_line_img = remove_dots(img)
    HP = np.sum(no_dots_line_img, 1).astype('int32')
    peak = np.amax(HP)
    return np.where(HP == peak)[0][0]


letters_dir = Path("/home/ramez/PycharmProjects/GradProject/letters")
letters = [letters_dir.joinpath(letter) for letter in letters_dir.iterdir()]
letters = [(letter.absolute(), cv2.threshold(cv2.cvtColor(cv2.imread(str(letter.absolute()), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
           for letter in letters]
# for (letter, img) in letters:
#     print(letter.name)
#     img = cv2.bitwise_not(img)
#     cv2.imwrite(letter.absolute().name, img)

# img = cv2.imread("/home/ramez/PycharmProjects/GradProject/Screenshot_20221212_105958.png", cv2.IMREAD_COLOR)
img = cv2.imread("/home/ramez/PycharmProjects/GradProject/IMG-20221212-WA0012.jpg", cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(_, binary_img) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binary_img = cv2.bitwise_not(binary_img)

# binary_img[baseline_idx: baseline_idx + 10, :] = 0
# cv2.imwrite("lol.png", binary_img)

baseline_idx = baseline(binary_img)

VP = np.sum(binary_img, 0).astype('int32')  # vertical projection

# y2 = normalized(VP, 0)
# heights = MAX_PLOT_HEIGHT * y2
# plt.plot(y)
# plt.show()

d2 = cv2.matchShapes(binary_img, binary_img[:, 350:], cv2.CONTOURS_MATCH_I2, 0)
# cv2.imshow("lol", binary_img[:][350:])
# cut = binary_img[:, 384:400]
# cut = binary_img[:, 312:328]
# cut = binary_img[:, 350:384]
# cut = binary_img[:, 328:350]
cut = binary_img[:, 290:312]
cv2.imwrite("cut.png", cut)
for letter, img in letters:
    print(str(letter), cv2.matchShapes(cut, img, cv2.CONTOURS_MATCH_I2, 0))

# line_img = cv2.imread("/home/ramez/PycharmProjects/GradProject/Screenshot_20221212_105958.png", cv2.IMREAD_GRAYSCALE)
# projection_bins = np.sum(line_img, 0).astype('int32')  # vertical projection
