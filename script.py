import cv2
import numpy as np
from Cell import Cell
from imutils import grab_contours
from matplotlib import pyplot as plt
from Transformation import Transformation
from tensorflow.python.keras.saving.save import load_model
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform
import os
from tensorflow import logging
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# image files
files = [
    'easy/sudoku1.png',
    'easy/sudoku2.png',
    'easy/sudoku3.png',
    'easy/sudoku4.png',
    'easy/sudoku5.png',
    'easy/sudoku6.png',
    'easy/sudoku7.png',
    'medium/sudoku1.png',
    'medium/sudoku2.png',
    'medium/sudoku3.png',
    'medium/sudoku4.png',
    'medium/sudoku5.jpg',
    'medium/sudoku6.jpg',
    'medium/sudoku7.jpg',
    'hard/sudoku1.jpg',
    'hard/sudoku2.png',
    'hard/sudoku3.png',
    'hard/sudoku4.png',
    'hard/sudoku5.png',
    'hard/sudoku6.png',
]


model = load_model('digits.h5')


def predict_digit(img):
    """Determine a digit from an image using CNN model"""

    resized_img = cv2.resize(img, (28, 28))
    np_img = np.array(resized_img)
    np_img = np_img.reshape(1, 28, 28, 1)
    np_img = np_img/255.0
    res = model.predict([np_img])[0]
    res = list(zip(range(0, 10), res))
    res.sort(key=lambda x: x[1])
    return res[-1][0] if res[-1][0] != 0 else res[-2][0]


def save_transformations(file_name, transformations, axes):
    """Save prepared transformations to result directory"""

    # plot all transformations
    for ax, transformation in zip(axes, transformations.values()):
        transformation.plot(ax)

    # save transformation steps
    plt.savefig(f'./results/{file_name}', dpi=400)
    plt.close()


for file in files:
    print(f'Starting: ./images/{file}')
    # load image
    bgr_image = cv2.imread(f'./images/{file}')

    # color conversions
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # transformations
    median = cv2.medianBlur(grayscale_image, 3)

    canny = cv2.Canny(median, 50, 220)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # find contours
    cv2_contours = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imutils_contours = sorted(grab_contours(cv2_contours),
                              key=cv2.contourArea, reverse=True)

    # check contour shape
    approximated_contours = [cv2.approxPolyDP(
        contour, 0.02 * cv2.arcLength(contour, True), True) for contour in imutils_contours]
    four_pointed_contours = [
        contour for contour in approximated_contours if len(contour) == 4]

    # if no rectangle found then algorithm failed
    if not len(four_pointed_contours):

        # image transformations for plotting
        transformations = {}

        # matplotlib plots
        px = 1 / plt.rcParams['figure.dpi']
        fig, axes = plt.subplots(
            figsize=(1920 * px, 1080 * px), nrows=2, ncols=3)
        axes = list(np.array(axes).flat)

        # load error image
        error = cv2.imread(
            f'./images/others/no_solution.png', cv2.IMREAD_GRAYSCALE)

        # prepare for plotting picture transformations
        transformations['rgb'] = Transformation(
            rgb_image, 'Original Image', False)
        transformations['grayscale'] = Transformation(
            grayscale_image, 'Grayscale', True)
        transformations['median'] = Transformation(median, 'Median Blur', True)
        transformations['canny'] = Transformation(
            canny, 'Canny Edge Detection', True)
        transformations['closing'] = Transformation(
            closing, 'Dilation followed by Erosion', True)
        transformations['error'] = Transformation(
            error, 'Status', True)

        save_transformations(file, transformations, axes)
    # if rectangle found then find solution
    else:
        # assume that the biggest rectangle is the puzzle border
        sudoku_border = four_pointed_contours[0]

        # display border on original image
        outlined = rgb_image.copy()
        cv2.drawContours(outlined, [sudoku_border], -1, (0, 255, 0), 2)

        # crop the original image using found border
        cropped = four_point_transform(
            rgb_image, sudoku_border.reshape(4, 2))

        # calculate the size of puzzle
        cropped_height, cropped_width, _cropped_channels = cropped.shape
        cell_height = cropped_height // 9
        cell_width = cropped_width // 9

        # divide puzzle into cells
        cells = [[Cell({'original': cropped[y * cell_height: (y + 1) * cell_height, x *
                                            cell_width: (x + 1) * cell_width]}, None, None) for x in range(9)] for y in range(9)]

        # find digit for each cell
        for row in cells:
            for cell in row:
                # transformations
                cell.image['grayscale'] = cv2.cvtColor(
                    cell.image['original'], cv2.COLOR_BGR2GRAY)
                cell.image['otsu'] = cv2.threshold(
                    cell.image['grayscale'], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                cell.image['inverted'] = cv2.bitwise_not(cell.image['otsu'])
                cell.image['no_borders'] = clear_border(cell.image['inverted'])
                cell.image['with_digit'] = np.zeros(
                    (cell_height, cell_width, 3), np.uint8)
                if cv2.countNonZero(cell.image['no_borders']) / float(cell_height * cell_width) >= 1 / 30:
                    # find the digit
                    cell.contains_digit = True
                    cell.digit = predict_digit(cell.image['no_borders'])
                    # draw the digit
                    x = int(cell_width * 0.33)
                    y = int(cell_height * 0.8)
                    cv2.putText(cell.image['with_digit'], str(
                        cell.digit), (x, y), cv2.FONT_HERSHEY_COMPLEX, min(cell_width, cell_height) / (25 / 0.5), (0, 255, 0), 2)
                else:
                    # no digit exists in the given cell
                    cell.contains_digit = False
                    cell.digit = 0

        # merge cells without borders
        merged = np.concatenate([np.concatenate(
            [cell.image['no_borders'] for cell in row], axis=1) for row in cells], axis=0)

        # board with found digits
        with_digits = np.concatenate([np.concatenate(
            [cell.image['with_digit'] for cell in row], axis=1) for row in cells], axis=0)

        # image transformations for plotting
        transformations = {}

        # matplotlib plots
        px = 1 / plt.rcParams['figure.dpi']
        fig, axes = plt.subplots(
            figsize=(1920 * px, 1080 * px), nrows=2, ncols=7)
        axes = list(np.array(axes).flat)

        # prepare for plotting picture transformations
        transformations['rgb'] = Transformation(
            rgb_image, 'Original Image', False)
        transformations['grayscale'] = Transformation(
            grayscale_image, 'Grayscale', True)
        transformations['median'] = Transformation(median, 'Median Blur', True)
        transformations['canny'] = Transformation(
            canny, 'Canny Edge Detection', True)
        transformations['closing'] = Transformation(
            closing, 'Dilation followed by Erosion', True)
        transformations['outlined'] = Transformation(
            outlined, 'Found Contour', False)
        transformations['cropped'] = Transformation(
            cropped, 'Cropped Image', False)

        # find cell with digit
        cell_position = [0, 0]

        for row_index, row in enumerate(cells):
            for cell_index, cell in enumerate(row):
                if cell.digit:
                    cell_position = [cell_index, row_index]

        # prepare for plotting cell transformation
        transformations['original_cell'] = Transformation(
            cells[cell_position[1]][cell_position[0]].image['original'], 'Extracted Cell', True)
        transformations['grayscale_cell'] = Transformation(
            cells[cell_position[1]][cell_position[0]].image['grayscale'], 'Grayscale', True)
        transformations['otsu_cell'] = Transformation(
            cells[cell_position[1]][cell_position[0]].image['otsu'], 'Otsu\'s Thresholding', True)
        transformations['inverted_cell'] = Transformation(
            cells[cell_position[1]][cell_position[0]].image['inverted'], 'Inverted', True)
        transformations['no_borders_cell'] = Transformation(
            cells[cell_position[1]][cell_position[0]].image['no_borders'], 'Without Borders', True)

        # prepare for plotting the end result
        transformations['concatenated_cells'] = Transformation(
            merged, 'Merged cells', True)
        transformations['with_numbers'] = Transformation(
            with_digits, 'End result', False)

        save_transformations(file, transformations, axes)
        print(f'Completed: ./images/{file}')
