import random

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
    return img[y1:y2, x1:x2,:]


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


def show_crop_fields(img,img2, intersections_sparse_matrix, shape=(1, 1), lines=np.array([])):
    """
    Crops fields from input image
    :param img: source input image
    :param intersections_sparse_matrix: sparse matrix with organised intersection points
    :param shape: shape of cropped chess field. Shape is measured in chess figure square
    :return: List of cropped images
    """
    def mask_lines (lines):
        edges = cv2.Canny(img, 100, 200)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=lines,
                                minLineLength=100, maxLineGap=80)



        if lines is not None:
            a, b, c = lines.shape
            for i in range(a):
                cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), [255,255,255], 10,
                         cv2.LINE_AA)

    mask_lines(lines)
    def draw_4_lines(img,left_upper_corner, left_lower_corner, right_lower_corner, right_upper_corner, color):
        cv2.line(img, left_upper_corner, left_lower_corner, color, 9, cv2.LINE_AA)
        cv2.line(img, left_upper_corner, right_upper_corner, color, 9, cv2.LINE_AA)
        cv2.line(img, right_lower_corner, right_upper_corner, color, 9, cv2.LINE_AA)
        cv2.line(img, right_lower_corner, left_lower_corner, color, 9, cv2.LINE_AA)
        return img

    def increase_brightness(img, value=30):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    res = []
    for i in range(len(intersections_sparse_matrix) - shape[0]):
        for j in range(len(intersections_sparse_matrix[i]) - shape[1]):
            left_upper_corner = intersections_sparse_matrix[i][j]
            left_lower_corner = intersections_sparse_matrix[i + shape[0]][j]
            right_lower_corner = intersections_sparse_matrix[i + shape[0]][j + shape[1]]
            right_upper_corner = intersections_sparse_matrix[i][j + shape[1]]

            if not np.any(np.isnan(left_upper_corner)) and not np.any(np.isnan(right_lower_corner)):
                left_upper_corner = left_upper_corner.astype(int)
                right_lower_corner = right_lower_corner.astype(int)
                cropped = crop_box(img, (*left_upper_corner, *right_lower_corner))
                cropped2 = crop_box(img2, (*left_upper_corner, *right_lower_corner))
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:


                    # Using cv2.blur() method
                    # alpha = 1  # Contrast control (1.0-3.0)
                    # beta = 0  # Brightness control (0-100)
                    # adjusted = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
                    # div = 32
                    # cropped = cropped // div * div + div // 2
                    # cropped = increase_brightness(cropped,value=30)
                    # gaussian_3 = cv2.GaussianBlur(cropped, (0, 0), 2.0)
                    # unsharp_image = cv2.addWeighted(cropped, 2.0, gaussian_3, -1.0, 0)
                    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    # gray = draw_4_lines(gray.copy(), left_upper_corner.astype(int), left_lower_corner.astype(int), right_lower_corner.astype(int), right_upper_corner.astype(int), 255)

                    # gray = (((gray) / 255) ** 2)*255
                    # gray = gray.astype(np.uint8)
                    kernel = np.array([[0, -1, 0],
                                       [-1, 7, -1],
                                       [0, -1, 0]])
                    image_1 = cv2.filter2D(src=cropped, ddepth=-1, kernel=kernel)
                    kernel = np.array([[0, -1, 0],
                                       [-1, 11, -1],
                                       [0, -1, 0]])
                    image_2 = cv2.filter2D(src=cropped, ddepth=-1, kernel=kernel)

                    # image = (((255 - image) / 255) ** 2)*255
                    gx = cv2.Sobel(cropped, cv2.CV_32F, 1, 0, ksize=1)
                    gy = cv2.Sobel(cropped, cv2.CV_32F, 0, 1, ksize=1)
                    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
                    mag = mag/mag.max()
                    image_3 = image_2 - image_2*mag
                    sift = cv2.SIFT_create(	nfeatures = 400,nOctaveLayers = 3,
                                               contrastThreshold = 0.05,edgeThreshold = 10
                                               ,sigma = 1.6)
                    kp = sift.detect(image_1.astype(np.uint8), None)
                    kp2 = sift.detect(image_2.astype(np.uint8), None)
                    kp3 = sift.detect(image_3.astype(np.uint8), None)
                    # imgq = cv2.drawKeypoints(cropped, kp, cropped)
                    x1, y1, x2, y2 = (*left_upper_corner, *right_lower_corner)
                    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                        img_, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)


                    # cv2.drawKeypoints(img[y1:y2, x1:x2, :], kp, img[y1:y2, x1:x2, :])
                    # cv2.drawKeypoints(img[y1:y2, x1:x2, :], kp2, img[y1:y2, x1:x2, :])
                    # cv2.drawKeypoints(img[y1:y2, x1:x2, :], kp3, img[y1:y2, x1:x2, :])

                    # cv2.drawKeypoints(img2[y1:y2, x1:x2], kp, img2[y1:y2, x1:x2])
                    cv2.drawKeypoints(img2[y1:y2, x1:x2], kp2, img2[y1:y2, x1:x2])
                    # cv2.drawKeypoints(img2[y1:y2, x1:x2], kp3, img2[y1:y2, x1:x2])


                    # minDist = 200
                    # param1 = 30  # 500
                    # param2 = 30  # 200 #smaller value-> more false circles
                    # minRadius = 5
                    # maxRadius = 100  # 10
                    # blurred = cv2.medianBlur(gray, 5)
                    # circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                    #                            minRadius=minRadius, maxRadius=maxRadius)
                    # cropped_ = cropped.copy()
                    # if circles is not None:
                    #     circles = np.uint16(np.around(circles))
                    #     for x in circles[0, :]:
                    #         cv2.circle(cropped, (x[0], x[1]), x[2], (0, 255, 0), 2)

                    # cv2.imshow("asd2", image)
                    # cv2.imshow("asd2e", imgq)
                    # cv2.imshow("cropped", cropped2)
                    # cv2.imshow("mag", mag/mag.max())
                    # cv2.imshow("angle", angle/angle.max())
                    # cv2.imshow("gray", gray.astype(np.uint8))
                    # cv2.imshow("img", cv2.resize(img, (1000,1000)))
                    # cv2.imshow("img2", cv2.resize(img2, (1000,1000)))
                    # cv2.waitKey()
                    tmp_img = draw_4_lines(img2.copy(), left_upper_corner.astype(int), left_lower_corner.astype(int), right_lower_corner.astype(int), right_upper_corner.astype(int), [0, 255, 0])
                    rate = (tmp_img.shape[0]) / cropped.shape[0]
                    stack = np.hstack((tmp_img, cv2.resize(cropped2,(int(cropped2.shape[1]*rate) , tmp_img.shape[0]))))
                    res.append(stack)
            else:
                break

                    #cv2.imwrite(r"D:\projekty\chessboard_checker_segmentation\dataset\cropped\images2\{}_{}.jpg".format(i,j), cropped)
                    # cv2.imwrite(r"D:\projekty\chessboard_checker_segmentation\dataset\cropped\masks\{}_{}.jpg".format(i,j), cropped2)
                    # img = cv2.rectangle(img, left_upper_corner, right_lower_corner, [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], 9)
                    #rate = (tmp_img.shape[0]) / cropped.shape[0]
                    #stack = np.hstack((tmp_img, cv2.resize(cropped,(int(cropped.shape[1]*rate) , tmp_img.shape[0]))))
                    #rate = (tmp_img.shape[0]//4) / cropped.shape[0]
                    # cv2.imshow("all", cv2.resize(stack, (int(stack.shape[1]//4) ,stack.shape[0]//4)))
                    # cv2.imwrite(r"D:\projekty\chessboard_checker_segmentation\resources\readme\bboxes" + r'\{}_{}.jpg'.format(i,j), cv2.resize(stack, (int(tmp_img.shape[1]) ,tmp_img.shape[0])))
                    # cv2.imshow("cropped", cv2.resize(cropped,(int(cropped.shape[1]*rate) , tmp_img.shape[0]//4)))
                    # cv2.waitKey(0)
        # sq = ((255-gray)/255)**2
        # sq2 = ((255-gray)/255)
        # sq = sq /sq.max()
        # cv2.imshow("asd", sq)
        # cv2.imshow("asd2", image)
        for i in range(len(res)):
            stack = res[i]
        cv2.imwrite(r"D:\projekty\chessboard_checker_segmentation\resources\readme\window_1_1.jpg".format(i),
                        img2)

    return res