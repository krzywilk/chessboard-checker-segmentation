import numpy as np
import cv2
import math


def create_lines(image, rho = 1, theta = np.pi / 180, threshold = 200):
    edges = auto_canny_filter(image)
    lines = cv2.HoughLines(edges, rho,theta, threshold)
    return lines


def auto_canny_filter(image : np.ndarray, sigma : int  = 0.33) -> np.ndarray:
    """src https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def filter_close_lines(lines : np.ndarray, atol_rho : int = 10, atol_theta : int = np.pi / 36) -> np.ndarray:
    res_lines_alloc = np.zeros([len(lines), 1, 2])
    res_lines_alloc[0] = lines[0] #the most confident
    end_alloc_pointer = 1
    for line in lines:
        rho, theta = line[0]
        if rho < 0:
            rho *= -1
            theta -= np.pi
        closeness_rho = np.isclose(rho, res_lines_alloc[0:end_alloc_pointer, 0, 0], atol=atol_rho)
        closeness_theta = np.isclose(theta, res_lines_alloc[0:end_alloc_pointer, 0, 1], atol=atol_theta)
        closeness = np.all([closeness_rho, closeness_theta], axis=0)
        if not any(closeness):
            res_lines_alloc[end_alloc_pointer] = line
            end_alloc_pointer += 1
    return res_lines_alloc[:end_alloc_pointer]



if __name__ == '__main__':
    img = cv2.imread("samp1.jpg")
    img = cv2.resize(img, (1500,1500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = create_lines(gray)
    lines = filter_close_lines(lines)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1500*(-b)), int(y0 + 1500*(a)))
            pt2 = (int(x0 - 1500*(-b)), int(y0 - 1500*(a)))
            cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    # if lines2 is not None:
    #     for i in range(0, len(lines2)):
    #         rho = lines2[i][0][0]
    #         theta = lines2[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1500*(-b)), int(y0 + 1500*(a)))
    #         pt2 = (int(x0 - 1500*(-b)), int(y0 - 1500*(a)))
    #         cv2.line(img, pt1, pt2, (0,255,255), 3, cv2.LINE_AA)
    cv2.imshow("x", img)
    cv2.waitKey(0)