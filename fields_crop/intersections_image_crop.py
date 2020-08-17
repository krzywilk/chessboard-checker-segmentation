import numpy as np
import cv2


def crop_box(img, bbox):
    """
    Crops given bbox from image
    :param img: source input image
    :param bbox: left upper corner and right bottom corner coords
    :return: cropped image part
    """
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    """
    Add extra space to the edges of image
    :param img: source input image
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return: padded to crop img
    """
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def crop_fields(img, intersections_sparse_matrix, shape=(1, 1)):
    """
    Crops fields from input image
    :param img: source input image
    :param intersections_sparse_matrix: sparse matrix with organised intersection points
    :param shape: shape of cropped chess field. Shape is measured in chess figure square
    :return: List of cropped images
    """
    res = []
    for i in range(len(intersections_sparse_matrix) - shape[0]):
        for j in range(len(intersections_sparse_matrix[i]) - shape[1]):
            left_upper_corner = intersections_sparse_matrix[i][j]
            right_lower_corner = intersections_sparse_matrix[i + shape[0]][j + shape[1]]
            if not np.any(np.isnan(left_upper_corner)) and not np.any(np.isnan(right_lower_corner)):
                left_upper_corner = left_upper_corner.astype(int)
                right_lower_corner = right_lower_corner.astype(int)
                cropped = crop_box(img, (*left_upper_corner, *right_lower_corner))
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    res.append(cropped)
            else:
                break
    return res
