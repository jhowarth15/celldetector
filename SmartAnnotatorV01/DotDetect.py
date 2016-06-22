from scipy.ndimage import *
import numpy as np
import cv2


class DotDetect():
    def __init__(self, probs):
        # pre process step: apply gaussian filter to probability array
        smoothed_probs = gaussian_filter(probs, 2)

        self.probs = smoothed_probs
        self.height, self.width = probs.shape

        self.cents = list()

    def detect_dots(self, kernel_size):
        dots = list()

        # get the local max points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = kernel.astype(bool)

        local_max = maximum_filter(self.probs, footprint=kernel)
        mask = (self.probs == local_max)
        local_max *= mask

        self.find_centroids(local_max)

        dot_regions = local_max.astype(bool).astype(np.uint8)

        contours, hierarchy = cv2.findContours(dot_regions, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            x, y = get_center_point(contour)
            probability = self.probs[y, x]
            dots.append(Dot(x, y, probability))

        return dots

    def find_centroids(self, local_max):
        for x in xrange(local_max.shape[0]):
            for y in xrange(local_max.shape[1]):
                self.cents.append((x,y))


def get_center_point(contour):
    x_vals = [row[0] for row in contour[:, 0]]
    y_vals = [row[1] for row in contour[:, 0]]
    min_x = min(x_vals)
    min_y = min(y_vals)
    max_x = max(x_vals)
    max_y = max(y_vals)
    cp_x = int(round((max_x + min_x) / 2))
    cp_y = int(round((max_y + min_y) / 2))
    return cp_x, cp_y


class Dot():
    def __init__(self, x, y, probability):
        self.x = x
        self.y = y
        self.probability = probability
