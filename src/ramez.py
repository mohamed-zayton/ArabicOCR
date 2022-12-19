import math

import numpy as np
import requests
import cv2
from pathlib import Path
from deskew import determine_skew
from character_segmentation import segment
from skimage.morphology import skeletonize, thin

def main(DEBUG=False):
    if DEBUG and Path("./output").exists():
        import shutil
        shutil.rmtree("./output")

    # Get image
    # image_url = "https://i.imgur.com/UEbXQRS.png"
    image_url = "https://i.imgur.com/0L8jTJ5.jpeg"
    # image_url = "https://i.imgur.com/qnO9Ef6.png"
    # image_url = "https://i.imgur.com/FMUzhkK.png"
    # image_url = "https://i.imgur.com/laxo7BT.png"
    # image_url = "https://i.imgur.com/l469BzW.png"
    # image_url = "https://i.imgur.com/uoWDvfT.jpg"
    # image_url = "https://imgs.search.brave.com/btK5Jee6ol01dLE2ApfuWBksCztG4Nw9TtWnE50Z_Mg/rs:fit:522:863:1/g:ce/aHR0cHM6Ly9zMy11/cy13ZXN0LTIuYW1h/em9uYXdzLmNvbS91/dy1zMy1jZG4vd3At/Y29udGVudC91cGxv/YWRzL3NpdGVzLzU0/LzIwMTYvMDQvMDYx/NDQzNDkvZ29vZC1z/Y2FuLnBuZw"
    # image_url = "https://i.imgur.com/fg5es4E.jpg"
    # image_url = "https://i.imgur.com/wxZu9kV.jpg"
    # image_url = "https://i.imgur.com/KSpvVux.jpg"
    # image_url = "https://i.imgur.com/2VyPlAN.jpg"
    # image_url = "https://i.imgur.com/R4KSuQc.png"
    # bottom-cropped-image: https://i.imgur.com/WNWYtJo.png
    resp = requests.get(image_url)
    orig_img = np.frombuffer(resp.content, dtype='uint8')
    orig_img = cv2.imdecode(orig_img, cv2.IMREAD_COLOR)
    # # orig_img = cv2.fastNlMeansDenoisingColored(orig_img, None, 10, 10, 7, 21)

    # orig_img = cv2.imread("/home/ramez/PycharmProjects/GradProject/Screenshot_20221116_130858.png")
    # orig_img = cv2.imread("/home/ramez/PycharmProjects/GradProject/Screenshot_20221116_131310.png")
    # orig_img = cv2.addWeighted(orig_img, 1.5, cv2.medianBlur(orig_img, 1), -0.5, 0.0)
    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = cv2.imwrite("output/output_origin_img.png", orig_img)
        print("Wrote original image" if write_successful else "Failed to write original image")

    # Binarization: grayscale -> invert colors -> threshold
    gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    # se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # bg = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, se)
    # gray_img = cv2.divide(gray_img, bg, scale=255)
    # binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    (_, binary_img) = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_img = cv2.bitwise_not(binary_img)

    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = cv2.imwrite("output/output_preprocessed_img.png", binary_img)
        print("Wrote preprocessed image" if write_successful else "Failed to write preprocessed image")

    # skeletonized_img = thin(binary_img // 255).astype(int) * 255
    # if DEBUG:
    #     Path("./output").mkdir(exist_ok=True)
    #     write_successful = cv2.imwrite("output/output_skeletonized_img.png", skeletonized_img)
    #     print("Wrote preprocessed image" if write_successful else "Failed to write preprocessed image")

    # Deskewing
    def rotate(image, angle, background):
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    angle = determine_skew(binary_img, angle_pm_90=True)
    # deskewed_img = rotate(binary_img, angle, (0, 0, 0))
    deskewed_img = binary_img

    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = cv2.imwrite("output/output_deskewed_img.png", deskewed_img)
        print("Wrote deskewed image" if write_successful else "Failed to write deskewed image")

    # Line splitting
    projection_bins = np.sum(deskewed_img, 1).astype('int32')  # horizontal projection
    empty_rows_above_line = 1
    empty_rows_below_line = 3

    consecutive_empty_columns = 0
    current_word_start = -1
    lines = []
    for idx, bin_ in enumerate(projection_bins):  # split image when consecutive empty lines are found
        if bin_ > (255 * 5):
            consecutive_empty_columns = 0
            if current_word_start == -1:
                current_word_start = idx
        elif current_word_start != -1:
            consecutive_empty_columns += 1
            if consecutive_empty_columns > empty_rows_below_line:
                lines.append(deskewed_img[max(current_word_start - empty_rows_above_line, 0):idx, :])
                consecutive_empty_columns = 0
                current_word_start = -1
    if current_word_start != -1:
        lines.append(deskewed_img[max(current_word_start - empty_rows_above_line, 0):, :])

    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = True
        for idx, line_img in enumerate(lines):
            write_successful = write_successful and cv2.imwrite(f"output/output_line_{idx}.png", line_img)
        print(f"Wrote {len(lines)} lines" if write_successful else f"Failed to write {len(lines)} lines")

    # Baseline detection
    baseline_dict = {}
    for line_img in lines:
        no_dots_line_img = remove_dots(line_img)
        HP = np.sum(no_dots_line_img, 1).astype('int32')
        peak = np.amax(HP)
        baseline_idx = np.where(HP == peak)[0]
        upper_base = baseline_idx[0]
        lower_base = baseline_idx[-1]
        thickness = abs(lower_base - upper_base) + 1

        line_img.flags.writeable = False

        baseline_dict[hash(line_img.data.tobytes())] = (upper_base, lower_base, thickness)

    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = True
        for idx, line_img in enumerate(lines):
            cpy = line_img.copy()
            upper_base, lower_base, thickness = baseline_dict[hash(line_img.data.tobytes())]
            cpy[upper_base : lower_base + 1, :] = 255
            write_successful = write_successful and cv2.imwrite(f"output/output_line_{idx}_baseline.png", cpy)
        print(f"Wrote {len(lines)} lines" if write_successful else f"Failed to write {len(lines)} lines")

    # Word splitting
    word_line_tuples = []
    for line_img in lines:
        projection_bins = np.sum(line_img, 0).astype('int32')  # vertical projection
        empty_columns_before_word = 1
        empty_columns_after_word = 1

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
                    words_in_line.append((line_img[:, max(current_word_start - empty_columns_before_word, 0):idx2], line_img))
                    consecutive_empty_columns = 0
                    current_word_start = -1
        if current_word_start != -1:
            words_in_line.append((line_img[:, max(current_word_start - empty_columns_before_word, 0):], line_img))

        for i in reversed(words_in_line):
            word_line_tuples.append(i)

    if DEBUG:
        Path("./output").mkdir(exist_ok=True)
        write_successful = True
        for idx, word_line in enumerate(word_line_tuples):
            write_successful = write_successful and cv2.imwrite(f"output/output_word_{idx}.png", word_line[0])
        print(f"Wrote {len(word_line_tuples)} words" if write_successful else f"Failed to write {len(word_line_tuples)} words")

    for idx, (word_img, line_img) in enumerate(word_line_tuples):
        img_chars = segment(line_img, word_img)
        # img_chars2 = [segment(line_img, img_char) for img_char in img_chars]
        # flat_list = [item for sublist in img_chars2 for item in sublist]

        if DEBUG:
            Path("./output").mkdir(exist_ok=True)

            for idx2, char_img in enumerate(img_chars):
                write_successful = cv2.imwrite(f"output/output_word_{idx}_char_{idx2}.png", char_img)
                if not write_successful:
                    print(f"Failed to write dotless word {idx} char {idx2}")


    # Character splitting
    for idx, (word_img, line_img) in enumerate(word_line_tuples):
        binary_word_img = word_img // 255  # the image was previously black-and-whited, so all values are 0 or 255

        no_dots_word_img = remove_dots(binary_word_img)

        if DEBUG:
            Path("./output").mkdir(exist_ok=True)
            write_successful = cv2.imwrite(f"output/output_word_{idx}_dotless.png", no_dots_word_img * 255)
            if not write_successful:
                print(f"Failed to write dotless word {idx}")

        # TODO wtf is this?
        VP_no_dots = np.sum(no_dots_word_img, 0).astype('int32')  # vertical projection
        filled_binary_word_img = binary_word_img.copy()
        (h, w) = filled_binary_word_img.shape
        for row in range(h - 1):
            for col in range(1, w - 1):

                if filled_binary_word_img[row][col] == 0 and filled_binary_word_img[row][col - 1] == 1 \
                        and filled_binary_word_img[row][col + 1] == 1 and filled_binary_word_img[row + 1][col] == 1 \
                        and VP_no_dots[col] != 0:
                    filled_binary_word_img[row][col] = 1

        if DEBUG:
            Path("./output").mkdir(exist_ok=True)
            write_successful = cv2.imwrite(f"output/output_word_{idx}_filled.png", filled_binary_word_img * 255)
            if not write_successful:
                print(f"Failed to write filled word {idx}")

        # TODO skeletonize

        upper_base, lower_base, MFV = baseline_dict[hash(line_img.data.tobytes())]

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


if __name__ == "__main__":
    main(DEBUG=True)
