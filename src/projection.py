import difflib
import multiprocessing
import os
import time
from functools import reduce

import cv2
import numpy as np
from pathlib import Path
import shutil
import copy

import scipy
import fastwer
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from sklearn.preprocessing import normalize
from skimage.feature import match_template, peak_local_max
from Levenshtein import distance as levenshtein_distance
from tqdm.auto import tqdm
from scipy.signal import fftconvolve, convolve2d


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


def resize_to_template(img, template_height = 30):
    HP = np.sum(remove_dots(img, img.shape[0] / 1.3), 1).astype('int32')
    HP_no_border = np.where(HP != 0)[0]
    height = HP_no_border[-1] - HP_no_border[0]
    return cv2.resize(img, None, fx=template_height / height, fy=template_height / height, interpolation=cv2.INTER_AREA)


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


def read_file_lines(filename):
    with open(filename) as file:
        return [line.rstrip() for line in file]


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


def normxcorr2(template, image):
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    out = convolve2d(image, np.flipud(np.fliplr(template)))

    # normalization
    image = convolve2d(np.square(image), a1) - np.square(convolve2d(image, a1)) / (np.prod(template.shape))
    image[np.where(image < 0)] = 0 # Remove small machine precision errors after subtraction
    out = out / np.sqrt(image * np.sum(np.square(template)))
    out[np.where(np.logical_not(np.isfinite(out)))] = 0 # Remove any divisions by 0 or very close to 0

    return out


def bestAlgo(img, letters, hard_thresh=0.70):
    cpy = copy.deepcopy(img)
    out = []
    while True:
        best_score = -1
        for letter, template in letters:
            res = cv2.matchTemplate(cpy, template, cv2.TM_CCOEFF_NORMED)
            max_ = res.max()
            if max_ >= best_score and max_ >= hard_thresh:
                loc = np.where(res >= max_)
                pt = next(zip(*loc[::-1]))
                best_score = max_
                w_, h_ = template.shape[::-1]
                (x, y, w, h, l) = (pt[0], pt[1], w_, h_, letter)

        if best_score < hard_thresh:
            break
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
resized_image = resize_to_template(binary_img, 34)

########################################################################## 0 ##########################################
# out = bestAlgo(resized_image, letters)
# for i, (x, y, w, h, letter) in enumerate(out):
#     cv2.imwrite(f"./output/letter_{i}.png", resized_image[y:y + h, x:x + w])
#
# print("".join([letter.name.replace("ـ", "").replace(".png", "") for (x, y, w, h, letter) in sorted(out, key=lambda box: box[0], reverse=True)]))
# for (x, y, w, h, letter) in out:
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (255, 255, 255))
# cv2.imwrite('./output/res.png', resized_image)

########################################################################## 0.2 ##########################################
# lines = list(zip(map(resize_to_template, split_into_lines(get_binarized_image("./data/7.png"))),
#             read_file_lines("./data/7.txt")))
#
# lines = lines + list(zip(map(resize_to_template, split_into_lines(get_binarized_image("./data/8.png"))),
#             read_file_lines("./data/8.txt")))

lines = list(zip(map(resize_to_template, split_into_lines(get_binarized_image("./Capture.png"))),
            ["أبجد هوز حطي كلمن سعفص قرشت ثخذ ضهلع"]))

def run(param, padding = 20):
    k, (line, ground_truth) = param
    line = np.pad(line, padding)
    words = split_into_words(line)
    word_predictions = []
    annotated_words = []
    for i, word in enumerate(words):
        word = np.pad(word, (20, 20))
        out = bestAlgo(word, letters)
        for j, (x, y, w, h, letter) in enumerate(sorted(out, key=lambda box: box[0], reverse=True)):
            cv2.imwrite(f"./output/line_{k}_word_{i}_letter_{j}.png", word[y:y + h, x:x + w])

        word_predictions.append("".join([letter.name.replace("ـ", "").replace(".png", "") for (x, y, w, h, letter) in
                                         sorted(out, key=lambda box: box[0], reverse=True)]))
        word_cpy = copy.deepcopy(word)
        for (x, y, w, h, letter) in out:
            cv2.rectangle(word_cpy, (x, y), (x + w, y + h), (255, 255, 255))
        annotated_words.append(word_cpy[padding:-padding, padding:-padding])
        cv2.imwrite(f"./output/line_{k}_word_{i}.png", word_cpy)

    annotated_line = reduce(lambda x,y: np.concatenate((x,y), axis=1), reversed(annotated_words))
    cv2.imwrite(f"./output/line_{k}.png", annotated_line)
    prediction = " ".join(word_predictions)
    cer = fastwer.score_sent(prediction, ground_truth, char_level=True)
    wer = fastwer.score_sent(prediction, ground_truth, char_level=False)
    return cer, wer


pool = multiprocessing.Pool(os.cpu_count())
res = pool.imap_unordered(run, enumerate(lines))
cers = []
wers = []
for x in tqdm(res, total=len(lines)):
    cers.append(x[0])
    wers.append(x[1])

print(f"CER: {np.mean(cers)}%")
print(f"WER: {np.mean(wers)}%")
print("Total characters: ", sum([len(i) for _, i in lines]))

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
# cv2.imwrite('./output/res.png', resized_image)
#
# for letter, template in [get_letter(letters, letter_name="ـفـ")]:
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.97)]
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
#                      detect_template(resized_image, template, hard_thresh=0.96)]
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
#                      detect_template(resized_image, template, hard_thresh=0.95)]
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
#                      detect_template(resized_image, template, hard_thresh=0.94)]
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
#                      detect_template(resized_image, template, hard_thresh=0.93)]
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
#                      detect_template(resized_image, template, hard_thresh=0.92)]
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
#                      detect_template(resized_image, template, hard_thresh=0.91)]
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
#                      detect_template(resized_image, template, hard_thresh=0.89)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res7.png', resized_image)
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
# cv2.imwrite('./output/res8.png', resized_image)
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
# cv2.imwrite('./output/res9.png', resized_image)
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
# cv2.imwrite('./output/res10.png', resized_image)
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
# cv2.imwrite('./output/res11.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.82)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res12.png', resized_image)
#
# for letter, template in letters:
#     w, h = template.shape[::-1]
#     boxes = boxes + [(box[0], box[1], w, h, letter) for box in
#                      detect_template(resized_image, template, hard_thresh=0.77)]
#
# for (x, y, w, h, letter) in boxes:
#     cv2.imwrite(f"./output/letter_{letter_counter}.png", resized_image[y:y + h, x:x + h])
#     letter_counter += 1
#     cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
# cv2.imwrite('./output/res13.png', resized_image)

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
# template = get_letter(letters, "كـ")[1]
# cv2.imwrite("./output/template.png", template)
# cv2.imwrite("./output/image.png", binary_img)
# result = normxcorr2(template, resized_image)
# y,x = np.unravel_index(result.argmax(), result.shape)
#
# fig = plt.figure(figsize=(8, 3))
# ax1 = plt.subplot(1, 3, 1)
# ax2 = plt.subplot(1, 3, 2)
# ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
#
# ax1.imshow(template, cmap=plt.cm.gray)
# ax1.set_axis_off()
# ax1.set_title('template')
#
# ax2.imshow(resized_image, cmap=plt.cm.gray)
# ax2.set_axis_off()
# ax2.set_title('image')
# rect = plt.Rectangle((x - template.shape[1], y - template.shape[0]), template.shape[1], template.shape[0], edgecolor='r', facecolor='none')
# ax2.add_patch(rect)
#
# ax3.imshow(normxcorr2(template, resized_image))
# ax3.set_axis_off()
# ax3.set_title('Cross Correlation')
# ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=5)
#
# plt.savefig('./output/match_template.png')

########################################################################## 5.0 ##########################################
# whole_img = get_binarized_image("./data/0.png")
# whole_img = whole_img[45:110, :]
#
#
# cv2.imwrite("./output/whole_img.png", whole_img)
#
# def get_line_cuts(img, empty_rows_above_line=1, empty_rows_below_line=1):
#     projection_bins = np.sum(img, 1).astype('int32')  # horizontal projection
#
#     consecutive_empty_columns = 0
#     current_line_start = -1
#     lines = []
#     for idx, bin_ in enumerate(projection_bins):  # split image when consecutive empty lines are found
#         if bin_ != 0:
#             consecutive_empty_columns = 0
#             if current_line_start == -1:
#                 current_line_start = idx
#         elif current_line_start != -1:
#             consecutive_empty_columns += 1
#             if consecutive_empty_columns > empty_rows_below_line:
#                 lines.append((max(current_line_start - empty_rows_above_line, 0), idx))
#                 consecutive_empty_columns = 0
#                 current_line_start = -1
#     if current_line_start != -1:
#         lines.append((max(current_line_start - empty_rows_above_line, 0), img.shape[0]))
#
#     return lines
#
# projection_bins = np.sum(whole_img, 1).astype('int32')  # horizontal projection
# x = [i for i in range(len(projection_bins))]
# plt.figure(figsize=(12, 12), dpi=200)
# plt.plot(projection_bins[::-1], x)
# plt.savefig("./output/hp.png")
#
# cuts = get_line_cuts(whole_img)
# for x1, x2 in cuts:
#     cv2.line(whole_img, (0, x1), (whole_img.shape[1], x1), (255, 255, 255))
#     cv2.line(whole_img, (0, x2), (whole_img.shape[1], x2), (255, 255, 255))
#     plt.axhline(y=x1, color='r', linestyle='-')
#     plt.axhline(y=x2, color='r', linestyle='-')
#
# plt.savefig("./output/hp_with_cuts.png")
# cv2.imwrite("./output/whole_img_with_cuts.png", whole_img)
#
# line = split_into_lines(get_binarized_image("./Capture.png"))[0]
# cv2.imwrite("./output/line.png", line)
# words = split_into_words(line, empty_columns_after_word=5)
# word = copy.deepcopy(words[4])
# cv2.imwrite("./output/word.png", word)
#
# def get_word_cuts(img, empty_columns_before_word=2, empty_columns_after_word=2):
#     projection_bins = np.sum(img, 0).astype('int32')  # vertical projection
#
#     consecutive_empty_columns = 0
#     current_word_start = -1
#     words_in_line = []
#     for idx2, bin_ in enumerate(projection_bins):  # split image when consecutive empty lines are found
#         if bin_ != 0:
#             consecutive_empty_columns = 0
#             if current_word_start == -1:
#                 current_word_start = idx2
#         elif current_word_start != -1:
#             consecutive_empty_columns += 1
#             if consecutive_empty_columns > empty_columns_after_word:
#                 words_in_line.append((max(current_word_start - empty_columns_before_word, 0), idx2))
#                 consecutive_empty_columns = 0
#                 current_word_start = -1
#     if current_word_start != -1:
#         words_in_line.append((max(current_word_start - empty_columns_before_word, 0), img.shape[1]))
#
#     return list(reversed(words_in_line))
#
# plt.figure(figsize=(20, 8), dpi=200)
# projection_bins = np.sum(line, 0).astype('int32')  # vertical projection
# plt.plot(projection_bins)
# plt.savefig("./output/vp_line.png")
#
# cuts = get_word_cuts(line)
# for x1, x2 in cuts:
#     cv2.line(line, (x1, 0), (x1, whole_img.shape[1]), (255, 255, 255))
#     cv2.line(line, (x2, 0), (x2, whole_img.shape[1]), (255, 255, 255))
#     plt.axvline(x=x1, color='r', linestyle='-')
#     plt.axvline(x=x2, color='r', linestyle='-')
#
# cv2.imwrite("./output/line_with_cuts.png", line)
# plt.savefig("./output/vp_line_with_cuts.png")
#
# plt.figure(figsize=(20, 8), dpi=200)
# projection_bins = np.sum(word, 0).astype('int32')  # vertical projection
# plt.plot(projection_bins)
# plt.savefig("./output/vp_word.png")

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
