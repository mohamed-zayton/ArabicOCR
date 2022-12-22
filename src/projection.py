import difflib
import time

import cv2
import numpy as np
from pathlib import Path
import shutil
import copy

import scipy
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from sklearn.preprocessing import normalize
from skimage.feature import match_template, peak_local_max
from Levenshtein import distance as levenshtein_distance
from tqdm.auto import tqdm


def cleanup():
    if Path("./output").exists():
        shutil.rmtree("./output")
    Path("./output").mkdir(exist_ok=True)


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


def load_image(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)


def resize_to_template(img):
    HP = np.sum(remove_dots(img, img.shape[1] / 2), 1).astype('int32')
    HP_no_border = np.where(HP != 0)[0]
    height = HP_no_border[-1] - HP_no_border[0]
    return cv2.resize(img, None, fx=30 / height, fy=30 / height, interpolation=cv2.INTER_AREA)


def get_binarized_image(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, binary_img) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_img = cv2.bitwise_not(binary_img)
    return binary_img


def split_into_lines(img, empty_rows_above_line=1, empty_rows_below_line=1):
    projection_bins = np.sum(img, 1).astype('int32')  # horizontal projection

    consecutive_empty_columns = 0
    current_line_start = -1
    lines = []
    for idx, bin_ in enumerate(projection_bins):  # split image when consecutive empty lines are found
        if bin_ != 0:
            consecutive_empty_columns = 0
            if current_line_start == -1:
                current_line_start = idx
        elif current_line_start != -1:
            consecutive_empty_columns += 1
            if consecutive_empty_columns > empty_rows_below_line:
                lines.append(img[max(current_line_start - empty_rows_above_line, 0):idx, :])
                consecutive_empty_columns = 0
                current_line_start = -1
    if current_line_start != -1:
        lines.append(img[max(current_line_start - empty_rows_above_line, 0):, :])

    return lines


def split_into_words(img, empty_columns_before_word=2, empty_columns_after_word=2):
    projection_bins = np.sum(img, 0).astype('int32')  # vertical projection

    consecutive_empty_columns = 0
    current_word_start = -1
    words_in_line = []
    for idx2, bin_ in enumerate(projection_bins):  # split image when consecutive empty lines are found
        if bin_ != 0:
            consecutive_empty_columns = 0
            if current_word_start == -1:
                current_word_start = idx2
        elif current_word_start != -1:
            consecutive_empty_columns += 1
            if consecutive_empty_columns > empty_columns_after_word:
                words_in_line.append(img[:, max(current_word_start - empty_columns_before_word, 0):idx2])
                consecutive_empty_columns = 0
                current_word_start = -1
    if current_word_start != -1:
        words_in_line.append(img[:, max(current_word_start - empty_columns_before_word, 0):])

    return list(reversed(words_in_line))


def load_letters(path="./Images"):
    letters_dir = Path(path)
    letters = filter(lambda letter: letter.name.endswith(".png"), letters_dir.iterdir())
    return [(letter.absolute(),
             split_into_words(cv2.cvtColor(load_image(str(letter.absolute())), cv2.COLOR_BGR2GRAY))[0]) for letter in
            letters]


def get_letter(letters, letter_name):
    return next(filter(lambda x: x[0].name == f"{letter_name}.png", letters))


def detect_template(img, template, hard_thresh=0.7):
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= max(hard_thresh, res.max() * 0.95))
    return list(zip(*loc[::-1]))


def diff_strings(a, b):
    output = []
    matcher = difflib.SequenceMatcher(None, a, b)
    green = '\x1b[38;5;16;48;5;2m'
    red = '\x1b[38;5;16;48;5;1m'
    endgreen = '\x1b[0m'
    endred = '\x1b[0m'

    width = 0
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == 'equal':
            output.append(a[a0:a1])
            width += a1 - a0
        elif opcode == 'insert':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
            width += b1 - b0
        elif opcode == 'delete':
            output.append(f'{red}{a[a0:a1]}{endred}')
            width += a1 - a0
        elif opcode == 'replace':
            output.append(f'{green}{b[b0:b1]}{endgreen}')
            output.append(f'{red}{a[a0:a1]}{endred}')
            width += a1 - a0
            width += b1 - b0
    return ''.join(output), width


def bestAlgo(img, letters, hard_thresh=0.70, granularity=0.001):
    cpy = copy.deepcopy(img)
    out = []
    thresh = 0.998
    while True:
        if thresh < hard_thresh:
            break

        (x, y, w, h, l) = None, None, None, None, None
        best_score = -1
        for letter, template in letters:
            res = cv2.matchTemplate(cpy, template, cv2.TM_CCOEFF_NORMED)
            if res.max() >= best_score and res.max() >= thresh:
                loc = np.where(res >= res.max())
                pt = next(zip(*loc[::-1]))
                best_score = res.max()
                w_, h_ = template.shape[::-1]
                (x, y, w, h, l) = (pt[0], pt[1], w_, h_, letter)

        if best_score == -1:
            thresh -= granularity
        else:
            out.append((x, y, w, h, l))
            cv2.rectangle(cpy, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)

    return out


cleanup()
letters = load_letters()

img = cv2.imread("Capture.png", cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(_, binary_img) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
binary_img = cv2.bitwise_not(binary_img)
resized_image = resize_to_template(binary_img)

########################################################################## 0 ##########################################
# out = bestAlgo(resized_image, letters)
# for i, (x, y, w, h, letter) in enumerate(out):
#     cv2.imwrite(f"./output/letter_{i}.png", resized_image[y:y + h, x:x + w])
#
# print("".join([letter.name.replace("ـ", "").replace(".png", "") for (x, y, w, h, letter) in sorted(out, key=lambda box: box[0], reverse=True)]))
# for (x, y, w, h, letter) in out:
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255))
# cv2.imwrite('./output/res.png', resized_image)

########################################################################## 0.1 ##########################################
lines = [(resize_to_template(split_into_lines(get_binarized_image("Capture.png"))[0]), "أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ")]
distances = []
for k, (line, ground_truth) in tqdm(enumerate(lines), desc="lines", leave=True):
    line = np.pad(line, 20)
    words = split_into_words(line)
    word_predictions = []
    for i, word in enumerate(tqdm(words, desc="words", leave=False)):
        word = np.pad(word, (20, 20))
        out = bestAlgo(word, letters)
        for j, (x, y, w, h, letter) in enumerate(sorted(out, key=lambda box: box[0], reverse=True)):
            cv2.imwrite(f"./output/line_{k}_word_{i}_letter_{j}.png", word[y:y + h, x:x + w])

        word_predictions.append("".join([letter.name.replace("ـ", "").replace(".png", "") for (x, y, w, h, letter) in
                                         sorted(out, key=lambda box: box[0], reverse=True)]))
        cpy = copy.deepcopy(word)
        for (x, y, w, h, letter) in out:
            cv2.rectangle(cpy, (x, y), (x + w, y + h), (255, 255, 255))
        cv2.imwrite(f"./output/line_{k}_word_{i}.png", cpy)

    prediction = " ".join(word_predictions)
    distance = levenshtein_distance(prediction, ground_truth)
    # diff = diff_strings(prediction, ground_truth)
    # print("Ground truth: ", ground_truth.ljust(diff[1] - 2))
    # print("Prediction: ", diff[0])
    # print("Distance: ", distance)
    distances.append((distance, len(ground_truth)))

print("Average distance per letter", sum([x / y for x, y in distances]) / len(distances))

########################################################################## 1 ##########################################
# letter_counter = 0
# boxes = []
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.7)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255))
# cv2.imwrite('./output/res.png', resized_image)

########################################################################## 2 ##########################################
# letter_counter = 0
# boxes = []
#
# # for letter, template in [get_letter(letters, letter_name="ـفـ")]:
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.91)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + w])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res0.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.89)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res1.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.88)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res2.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.87)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res3.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.86)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res4.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.85)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res5.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.83)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res6.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.76)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res7.png', resized_image)


########################################################################## 2.1 ##########################################
# boxes = []
# # for letter, template in [get_letter(letters, letter_name="ـلـ")]:
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.90)]
#
# for i, (x, y, w, h, letter) in enumerate(boxes):
#     cv2.imwrite(f"./output/letter_{i}.png", resized_image[y:y + h, x:x + h])
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255))
# cv2.imwrite('./output/res.png', resized_image)

########################################################################## 3 ##########################################
# t = time.time()
# start = binary_img.shape[1]  # The whole width. This is our starting point because Arabic is RTL.
# counter = 0
# THRESH = 1e-4
# MIN_DISTANCE = 6
# out = []
# for _ in range(15):
#     best_end = start
#     best_distance = 10000000
#     best_cut = None
#     best_letter = ""
#     for i in range(start - 1, -1, -1):
#         cut = binary_img[:, i:start]
#         for letter, img in letters:
#             distance = cv2.matchShapes(cut, img, cv2.CONTOURS_MATCH_I2, 0)
#             if distance < best_distance and MIN_DISTANCE < start - i:
#                 best_distance = distance
#                 best_cut = cut
#                 best_end = i
#                 best_letter = letter.name
#
#     cv2.imwrite(f"./output/{counter}.png", best_cut)
#     out.append(best_end)
#     start = best_end
#     counter = counter + 1
# print(time.time() - t)
#
# for i in out:
#     cv2.line(binary_img, (i, 0), (i, binary_img.shape[0]), (255, 255, 255))
# cv2.imwrite(f"./output/res.png", binary_img)

########################################################################## 4 ##########################################
# coin = get_letter(letters, "كـ")[1]
# result = match_template(resized_image, coin)
# peaks = peak_local_max(result, min_distance=10, threshold_rel=0.5)
# ij = np.unravel_index(np.argmax(result), result.shape)
# x, y = ij[::-1]
#
# fig = plt.figure(figsize=(8, 3))
# ax1 = plt.subplot(1, 3, 1)
# ax2 = plt.subplot(1, 3, 2)
# ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
#
# ax1.imshow(coin, cmap=plt.cm.gray)
# ax1.set_axis_off()
# ax1.set_title('template')
#
# ax2.imshow(resized_image, cmap=plt.cm.gray)
# ax2.set_axis_off()
# ax2.set_title('image')
# # highlight matched region
# hcoin, wcoin = coin.shape
# rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
# ax2.add_patch(rect)
#
# ax3.imshow(result)
# ax3.plot(peaks[:, 1], peaks[:, 0], 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
# ax3.set_axis_off()
# ax3.set_title('`match_template`\nresult')
# # highlight matched region
# ax3.autoscale(False)
# ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#
# plt.savefig('./output/match_template.png')

########################################################################## 5 ##########################################
# baseline_idx = baseline(binary_img)
# binary_img[baseline_idx: baseline_idx + 3, :] = 255
# cv2.imwrite("./output/img_with_baseline.png", binary_img)
# MAX_PLOT_HEIGHT = 100
# VP = np.sum(binary_img, 0).astype('int32')  # vertical projection
# heights = VP / np.linalg.norm(VP) * MAX_PLOT_HEIGHT
# plt.plot(heights)
# plt.savefig('./output/vertical_projection.png')
# binary_img[baseline_idx: baseline_idx + 10, :] = 0
# MAX_PLOT_HEIGHT = 100
# VP = np.sum(binary_img, 0).astype('int32')  # vertical projection
# heights = VP / np.linalg.norm(VP) * MAX_PLOT_HEIGHT
# plt.figure()
# plt.plot(heights)
# plt.savefig('./output/vertical_projection_with_baseline_removed.png')

########################################################################## 6 ##########################################
# cut = binary_img[:, 877:907]
# cv2.imwrite("./output/cut.png", cut)
# vals = sorted([(letter, img, cv2.matchShapes(cut, img, cv2.CONTOURS_MATCH_I1, 0)) for letter, img in letters], key=lambda x: x[2], reverse=False)
# for (letter, img, score) in vals:
#     print(letter.name, score)
