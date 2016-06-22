__author__ = 'Daniele'

from Tkinter import Button, Label, Toplevel, PanedWindow, HORIZONTAL
import numpy as np
from PIL import Image, ImageTk
import cv2


class Refiner(Toplevel):
    def __init__(self, master, smart_annotator):
        Toplevel.__init__(self, master)

        self.void_image = np.zeros((100, 100, 3))

        self.smart_annotator = smart_annotator
        self.current_point = None
        self.index_frame = None

        self.pos_data = list()
        self.neg_data = list()

        self.uncertain_dots = list()

        self.current_point_img_array = Image.fromarray(self.void_image, "RGB")
        self.current_point_img = ImageTk.PhotoImage(self.current_point_img_array)

        img_label = Label(self, image=self.current_point_img)
        img_label.pack()

        button_paned_window = PanedWindow(self, orient=HORIZONTAL)
        button_paned_window.pack()

        self.probability_label = Label(self, text="0.0")
        button_paned_window.add(self.probability_label)

        yes_but = Button(self, text="Y", command=self.yes_command)
        button_paned_window.add(yes_but)

        no_but = Button(self, text="N", command=self.no_command)
        button_paned_window.add(no_but)

        maybe_but = Button(self, text="?", command=self.maybe_command)
        button_paned_window.add(maybe_but)

        self.protocol('WM_DELETE_WINDOW', self.intercept_close_action)

    def intercept_close_action(self):
        self.smart_annotator.update_datasets(self.pos_data, self.neg_data)
        self.withdraw()

    def show_image(self, image):
        self.current_point_img_array = Image.fromarray(image, "RGB")
        self.current_point_img.paste(self.current_point_img_array)

    def start(self, list_dots, low_thresh, high_thresh):
        list_uncertain_dots = list()

        for idx, dots in list_dots:
            for dot in dots:
                if (dot.probability > low_thresh) & (dot.probability < high_thresh):
                    list_uncertain_dots.append((idx, dot))

        self.uncertain_dots = rank_points_with_variance(list_uncertain_dots, self.smart_annotator)

        self.step()

    def step(self):
        # pop point from list of ranked uncertain dots
        self.index_frame, self.current_point = self.pop_element()

        # if the list is empty, display empty image
        if self.current_point is None:
            image = self.void_image

            self.probability_label.config(text='End.')
        else:
            # retrieve image from the point
            image = self.retrieve_image_from_point(self.current_point, self.index_frame)

            # magnify the image
            image = cv2.resize(image, (100, 100))

        # display the magnified image
        self.show_image(image)

    def pop_element(self):
        if len(self.uncertain_dots) == 0:
            return None, None

        idx, dot, variance = self.uncertain_dots.pop()
        point = (dot.y, dot.x)

        # updates probability label
        self.probability_label.config(text=str(round(dot.probability, 2)))

        # updates variance label

        return (idx) , (point)

    def yes_command(self):
        # add the point to the positive points list
        if self.current_point is not None:
            self.pos_data.append([self.current_point, self.index_frame])

        # step to the next point in the ranked uncertain list
        self.step()

    def no_command(self):
        # add the point to the negative points list
        if self.current_point is not None:
            self.neg_data.append([self.current_point, self.index_frame])

        # step to the next point in the ranked uncertain list
        self.step()

    def maybe_command(self):
        # step to the next point in the ranked uncertain list
        self.step()

    def retrieve_image_from_point(self, point, index):
        origin_array = self.smart_annotator.get_image_from_idx(index)
        patch_size = self.smart_annotator.settings.get_patch_size()
        imm = getPatchValues(point[1], point[0], patch_size, origin_array)

        self.smart_annotator.display_point(point, index)

        return imm


def near_in_list(point, list_, min_dist):
    for el in list_:
        dist = np.linalg.norm(np.array(point) - np.array(el))
        if dist < min_dist:
            return True
    return False

def rank_points_with_variance(uncertain_points, smart_annotator):
    ranked_points = list()
    variance_points = list()

    trees = smart_annotator.clf.estimators_
    image_feature = smart_annotator.image_feature
    image_array = None

    current_index = 0

    for idx, point in uncertain_points:
        if idx != current_index:
            image_array = smart_annotator.get_image_from_idx(idx)
            image_feature.update_features(image_array, idx, True)
            current_index = idx

        proba_array = np.zeros((len(trees), ))
        feats = image_feature.extractFeatsFromPoint((point.y, point.x), smart_annotator.settings.get_selection_mask())

        for i in xrange(len(trees)):
            proba = trees[i].predict_proba(feats)[0][1]
            proba_array[i] = proba

        variance_forest = np.var(proba_array)
        variance_points.append(variance_forest)

    indices_sorted_variances = np.argsort(variance_points)
    for i in xrange(len(indices_sorted_variances)):
        ranked_points.append(uncertain_points[indices_sorted_variances[i]] + (variance_points[indices_sorted_variances[i]], ))

    return ranked_points

#TODO think something nicer
def getPatchValues(x, y, offset, image_array):

    patchXliminf = max(0, x - offset)
    patchYliminf = max(0, y - offset)
    patchXlimsup = min(512, x + offset)
    patchYlimsup = min(512, y + offset)
    values = image_array[patchYliminf:patchYlimsup, patchXliminf:patchXlimsup, :]
    return values


