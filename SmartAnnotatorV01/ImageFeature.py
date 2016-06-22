import cv2
import scipy.ndimage as nd
import numpy as np


class ImageFeature:
    def __init__(self, patch_size):
        self.patch_size = patch_size

        self.feats_dictionary = {}

    def update_features(self, img_array, index, use_memory):

        if use_memory and (index in self.feats_dictionary):
            self.integral_img, self.avg_rc, self.avg_gc, self.avg_bc, self.avg_rc_h, \
            self.avg_gc_h, self.avg_bc_h, self.gauss1rc, self.gauss1gc, self.gauss1bc,\
            self.gauss35rc,self.gauss35gc, self.gauss35bc, self.log2rc, self.log2gc, self.log2bc,\
            self.log35rc, self.log35gc, self.log35bc = self.feats_dictionary[index]
            return

        img_array = mirror_borders(img_array, self.patch_size // 2)

        # integral image
        self.integral_img = cv2.integral(img_array[:,:,0])

        # average image red and green channel patch size
        self.avg_rc = cv2.blur(img_array[:,:,0], (self.patch_size, self.patch_size))
        self.avg_gc = cv2.blur(img_array[:,:,1], (self.patch_size, self.patch_size))
        self.avg_bc = cv2.blur(img_array[:,:,2], (self.patch_size, self.patch_size))

        # average images all three channels
        self.avg_rc_h = cv2.blur(img_array[:,:,0], (self.patch_size//2, self.patch_size//2))
        self.avg_gc_h = cv2.blur(img_array[:,:,1], (self.patch_size//2, self.patch_size//2))
        self.avg_bc_h = cv2.blur(img_array[:,:,2], (self.patch_size//2, self.patch_size//2))

        # gassiuan smoothed sigma 1
        self.gauss1rc = nd.gaussian_filter(img_array[:,:,0], 1)
        self.gauss1gc = nd.gaussian_filter(img_array[:,:,1], 1)
        self.gauss1bc = nd.gaussian_filter(img_array[:,:,2], 1)

        # gaussian smoothed sigma 3.5
        self.gauss35rc = nd.gaussian_filter(img_array[:, :, 0], 3.5)
        self.gauss35gc = nd.gaussian_filter(img_array[:, :, 1], 3.5)
        self.gauss35bc = nd.gaussian_filter(img_array[:, :, 2], 3.5)

        # laplace of gaussian sigma 2 (all three chaannels)
        self.log2rc = nd.gaussian_laplace(img_array[:,:,0], 2)
        self.log2gc = nd.gaussian_laplace(img_array[:,:,1], 2)
        self.log2bc = nd.gaussian_laplace(img_array[:,:,2], 2)

        # laplace of gaussian sigma 3.5
        self.log35rc = nd.gaussian_laplace(img_array[:,:,0], 3.5)
        self.log35gc = nd.gaussian_laplace(img_array[:,:,1], 3.5)
        self.log35bc = nd.gaussian_laplace(img_array[:,:,2], 3.5)

        if use_memory:
            # add the computed features to the dictionary
            self.feats_dictionary[index] = self.integral_img, self.avg_rc, self.avg_gc, self.avg_bc, self.avg_rc_h,\
                self.avg_gc_h, self.avg_bc_h, self.gauss1rc, self.gauss1gc, self.gauss1bc, self.gauss35rc, self.gauss35gc,\
                self.gauss35bc, self.log2rc, self.log2gc, self.log2bc, self.log35rc, self.log35gc, self.log35bc

    def merge_dictionaries(self, dictionary):
        self.feats_dictionary = dict(self.feats_dictionary.items() + dictionary.items())

    def avg_features(self, point):
        return np.array([self.avg_rc[point], self.avg_gc[point], self.avg_bc[point],
                self.avg_rc_h[point], self.avg_gc_h[point], self.avg_bc_h[point]])

    def gauss_features(self, point):
        return np.array([self.gauss1rc[point], self.gauss1gc[point], self.gauss1gc[point],
                self.gauss35rc[point], self.gauss35gc[point], self.gauss35bc[point]])

    def logs_features(self, point):
        return np.array([self.log2rc[point], self.log2gc[point], self.log2bc[point],
                self.log35rc[point], self.log35gc[point], self.log35bc[point]])

    def haar_like_features(self, point, patch_size):
        feats = np.zeros((6,))
        half_patch = patch_size // 2
        third_patch = patch_size // 3

        x = point[0]-half_patch
        y = point[1]-half_patch

        integimg = self.integral_img[x:(x+patch_size+1), y:(y+patch_size+1)]

        p0 = integimg[0, 0]
        p1 = integimg[third_patch, 0]
        p2 = integimg[half_patch, 0]
        p3 = integimg[patch_size - third_patch, 0]
        p4 = integimg[patch_size, 0]
        p5 = integimg[0, third_patch]
        p6 = integimg[patch_size, third_patch]
        p7 = integimg[0, half_patch]
        p8 = integimg[half_patch, half_patch]
        p9 = integimg[patch_size, half_patch]
        p10 = integimg[0, patch_size - third_patch]
        p11 = integimg[patch_size, patch_size - third_patch]
        p12 = integimg[0, patch_size]
        p13 = integimg[third_patch, patch_size]
        p14 = integimg[half_patch, patch_size]
        p15 = integimg[patch_size - third_patch, patch_size]
        p16 = integimg[patch_size, patch_size]

        # feature number 0
        gray = p9 + p0 - (p4 + p7)
        white = p16 + p7 - (p9 + p12)
        feats[0] = (gray - white)

        # feature number 1
        gray = p14 + p0 - (p2 + p12)
        white = p16 + p2 - (p4 + p14)
        feats[1] = (gray - white)

        # feature number 2
        gray = (p13 + p0 - (p1 + p12)) + (p16 + p3 - (p4 + p15))
        white = (p15 + p1 - (p3 + p13))
        feats[2] = (gray - white)

        # feature number 3
        gray = (p6 + p0 - (p4 + p5)) + (p16 + p10 - (p11 + p12))
        white = p11 + p5 - (p6 + p10)
        feats[3] = (gray - white)

        # feature number 4
        gray = (p8 + p0 - (p2 + p7)) + (p16 + p8 - (p9 + p14))
        white = (p9 + p2 - (p4 + p8)) + (p14 + p7 - (p8 + p12))
        feats[4] = (gray - white)

        # feature number 5
        p17 = integimg[third_patch, third_patch]
        p18 = integimg[patch_size - third_patch, third_patch]
        p19 = integimg[third_patch, patch_size - third_patch]
        p20 = integimg[patch_size - third_patch, patch_size - third_patch]
        white = p20 + p17 - (p18 + p19)
        gray = (p16 + p0 - (p4 + p12)) - white
        feats[5] = (gray - white)

        return feats

    def extractFeatsFromPoint(self, point, selection_mask):
        half_p = self.patch_size // 2
        point = (point[0] + half_p, point[1] + half_p)

        avg_mask = selection_mask[0:6]
        gauss_mask = selection_mask[6:12]
        logs_mask = selection_mask[12:18]
        haar_mask = selection_mask[18:24]
        return list(self.avg_features(point)[avg_mask]) +\
               list(self.gauss_features(point)[gauss_mask]) +\
               list(self.logs_features(point)[logs_mask]) +\
               list(self.haar_like_features(point, self.patch_size)[haar_mask])


def mirror_borders(image_array, pad):
    return cv2.copyMakeBorder(image_array, pad, pad, pad, pad, cv2.BORDER_REPLICATE)