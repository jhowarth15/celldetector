__author__ = 'Daniele'

import numpy as np
import multiprocessing as mp
import itertools
import cv2
import ImageFeature as If
# import ttk
# from Tkinter import *
from PIL import Image, ImageTk, ImageDraw
from sklearn import ensemble, utils
from scipy import misc, ndimage
import random
import matplotlib.pyplot as plt
import Refiner as Ref
import DotDetect as Dd
import SettingWindow as Sw
import ViterbiTracker as Vt
import Plotter3D as P3D
import time
import datetime
import sys

MAX_NUM_FEATURES = 24


class SmartAnnotator(object):
    def __init__(self, folder_path, num_frames, num_channels):
        self.folder_path = folder_path
        self.num_frames = num_frames

        # initialize the list of list of dots for already tested frames
        self.already_tested = [[] for i in range(num_frames + 1)]

        # initialize the two lists positive and negative dataset
        self.positive_dataset = list()
        self.negative_dataset = list()

        # initialize the occupied grid
        self.occupied_grid = np.zeros((512, 512), dtype='bool')

        # initialize the two async results objects
        self.res_plus = None
        self.res_minus = None

        # initialize the random forest classifier with default values
        self.clf = ensemble.RandomForestClassifier(300, max_features=0.5, max_depth=2, n_jobs=1)

        # initialize the probability map array
        self.probabilities = np.zeros((512, 512))

        # the sequence starts with the index 1
        self.current_idx = 1

        # create the window
        # self.root = Tk()
        # self.root.title("SmartAnnotator")
        # self.root.geometry('512x555')

        # self.root.bind("<Left>", self._on_left)
        # self.root.bind("<Right>", self._on_right)

        # create the refiner
        # self.refiner = Ref.Refiner(self.root, self)
        # self.refiner.withdraw()

        # take a random image for the mser setting tab
        mser_image = self.get_image_from_idx(random.randint(1, num_frames-1))

        # create the settings window
        # self.settings = Sw.SettingWindow(self.root, MAX_NUM_FEATURES, self.num_frames, mser_image)
        # self.settings.withdraw()

        # buttons
        # button_paned_window = PanedWindow(orient=HORIZONTAL)
        # button_paned_window.grid(row=0, column=0)

        # self.settings_icon = ImageTk.PhotoImage(Image.open("icons/settings.png"))
        # self.settings_button = Button(self.root, image=self.settings_icon, command=self.settings.deiconify)
        # button_paned_window.add(self.settings_button)

        # self.combobox_value = StringVar()
        # combobox = ttk.Combobox(self.root, textvariable=self.combobox_value,
        #                         state='readonly', width=4)
        # list_channels = list()
        # for i in range(1, num_channels + 1):
        #     list_channels.append('ch' + str(i))
        # list_values = ['RGB'] + list_channels
        # combobox['values'] = tuple(list_values)
        # combobox.current(0)
        # combobox.bind("<<ComboboxSelected>>", self._new_combobox_selection)
        # button_paned_window.add(combobox)

        # train_forest_button = Button(self.root, text="Train", command=self.train_command)
        # button_paned_window.add(train_forest_button)

        # test_forest_button = Button(self.root, text="Test", command=self.test_command)
        # button_paned_window.add(test_forest_button)

        # add the confidence slider
        # self.slider = Scale(self.root, from_=0.0, to=1, resolution=0.01, orient=HORIZONTAL)
        # self.slider.set(0.7)
        # self.slider.bind("<ButtonRelease-1>", self.slider_command)
        # button_paned_window.add(self.slider)

        # self.overlay_button = IntVar()
        # check_button = Checkbutton(self.root, text="", variable=self.overlay_button, command=self.overlay)
        # button_paned_window.add(check_button)

        # refine_button = Button(self.root, text="Refine!", command=self.refine_command)
        # button_paned_window.add(refine_button)

        # left_button = Button(self.root, text="<", command=self.left_command)
        # button_paned_window.add(left_button)

        # right_button = Button(self.root, text=">", command=self.right_command)
        # button_paned_window.add(right_button)

        # self.current_idx_entry = Entry(self.root, width=5, justify=RIGHT)
        # self.current_idx_entry.bind("<Return>", self._return_on_entry)
        # self.current_idx_entry.bind("<ButtonRelease-1>", self._focus_on_entry)
        # self.current_idx_entry.insert(END, str(self.current_idx))
        # button_paned_window.add(self.current_idx_entry)

        # num_frames_label = Label(self.root, text='/' + str(self.num_frames))
        # button_paned_window.add(num_frames_label)

        # track_button = Button(self.root, text="Track", command=self.track_command)
        # button_paned_window.add(track_button)

        # image label
        self.imgArray = self.get_image_from_idx(self.current_idx)
        self.current_image = Image.fromarray(self.imgArray)
        # self.img = ImageTk.PhotoImage(self.current_image)

        # img_label = Label(self.root, image=self.img)
        # img_label.grid(row=1, column=0)

        # bind the click actions to the image label
        # img_label.bind("<Button 1>", self.add_positive_sample)
        # img_label.bind("<Button 2>", self.add_bunch_negative_samples)
        # img_label.bind("<Button 3>", self.add_negative_sample_event)

        # create the feature object and initialize it
        self.image_feature = If.ImageFeature(self.settings.get_patch_size())
        self.image_feature.update_features(self.imgArray, self.current_idx, True)

        # initialize multiprocessing pool
        self.pool = mp.Pool(processes=2)

        # flags
        self.updated = False
        self.has_been_tested = False

        self.root.mainloop()

    # ADD X AND Y HERE INSTEAD OF EVENT /////////////////////////////
    def add_positive_sample(self, event):
        # if in overlay mode discard
        # if self.overlay_button.get():
        #     return

        # disable feature tab in settings window
        # self.settings.notebook.tab(0, state='disabled')

        # calculate the center of mass
        x, y = self.get_center_of_mass(event.x, event.y, self.settings.get_patch_size())

        # check if features have been updated
        if not self.updated:
            # update the features
            self.image_feature.update_features(self.imgArray, self.current_idx, True)
            self.updated = True

        # get features from point and append it to positive dataset
        feats = self.image_feature.extractFeatsFromPoint((y, x), self.settings.get_selection_mask())
        self.positive_dataset.append(feats)

        # calculate patch coordinates
        p0, p1, p2, p3 = get_coordinates(x, y, self.settings.get_patch_size())

        # draw red rectangle
        # draw = ImageDraw.Draw(self.current_image)
        # draw.rectangle([(p0, p1), (p2, p3)], outline="red")
        # self.img.paste(self.current_image)

        # add extended patch to the occupied grid
        extended_patch = get_patch_coordinates(x, y, self.settings.get_patch_size()*2)
        for i in extended_patch:
            self.occupied_grid[i[1], i[0]] = True

    def add_negative_sample_event(self, event):
        x = event.x
        y = event.y

        self.add_negative_sample(x, y)

    def add_negative_sample(self, x, y):
        # disable feature tab in setting window
        self.settings.notebook.tab(0, state='disabled')

        # calculate patch coordinates
        patch = get_patch_coordinates(x, y, self.settings.get_patch_size())

        # check if features have been updated
        if not self.updated:
            # update the features
            self.image_feature.update_features(self.imgArray, self.current_idx, True)
            self.updated = True

        # get around 10 points per patch
        points = patch[1:-1:50]
        for point in points:
            # get features from point and append it to negative dataset
            feats = self.image_feature.extractFeatsFromPoint((point[1], point[0]), self.settings.get_selection_mask())
            self.negative_dataset.append(feats)

        # draw blue rectangle
        # draw = ImageDraw.Draw(self.current_image)
        # draw.rectangle([patch[0], patch[-1]], outline="blue")
        # self.img.paste(self.current_image)

    def add_bunch_negative_samples(self, event):
        num_neg = len(self.positive_dataset)*2
        for i in xrange(num_neg):
            randX = random.randrange(0, 512)
            randY = random.randrange(0, 512)
            while self.occupied_grid[randY, randX] is True:
                randX = random.randrange(0, 512)
                randY = random.randrange(0, 512)

            self.add_negative_sample(randX, randY)

    # def left_command(self):
    #     self.current_idx -= 1
    #     self.imgArray = self.get_image_from_idx(self.current_idx)
    #     self.current_idx_entry.delete(0, END)
    #     self.current_idx_entry.insert(END, self.current_idx)

    #     # the features have changed
    #     self.updated = False

    #     # update the frame
    #     self.current_image = Image.fromarray(self.select_channel(self.imgArray))
    #     self.img.paste(self.current_image)

    #     if self.res_minus is not None:
    #         idx, dots, self.probabilities, dictionary = self.res_minus.get()
    #         self.image_feature.merge_dictionaries(dictionary)
    #         self.already_tested[idx] = dots
    #         self._test_previous_frame()

    #     dots = self.already_tested[self.current_idx]
    #     if len(dots) > 0:
    #         self.show_crosses(dots, self.slider.get())

    # def right_command(self):
    #     self.current_idx += 1
    #     self.imgArray = self.get_image_from_idx(self.current_idx)
    #     self.current_idx_entry.delete(0, END)
    #     self.current_idx_entry.insert(END, self.current_idx)

    #     self.updated = False

    #     # update the frame
    #     self.current_image = Image.fromarray(self.select_channel(self.imgArray))
    #     self.img.paste(self.current_image)

    #     if self.res_plus is not None:
    #         idx, dots, self.probabilities, dictionary = self.res_plus.get()
    #         self.image_feature.merge_dictionaries(dictionary)
    #         self.already_tested[idx] = dots
    #         self._test_next_frame()

    #     dots = self.already_tested[self.current_idx]
    #     if len(dots) > 0:
    #         self.show_crosses(dots, self.slider.get())

    def train_command(self):
        # remove previously founded dots
        self.already_tested = [[] for i in range(self.num_frames + 1)]

        self.has_been_tested = False

        # cancel results asyncronous calls
        self.res_plus = None
        self.res_minus = None

        # merge the two datasets and create X and y
        X, y = merge_datasets(self.positive_dataset, self.negative_dataset)

        # updates the classifier object using the forest opts specified in the setting window
        # disable corresponding frame
        forest_opts = self.settings.get_forest_opts()
        self.clf = ensemble.RandomForestClassifier(forest_opts[0], max_depth=forest_opts[1], max_features=forest_opts[2])
        self.settings.notebook.tab(1, state='disabled')

        print(self.clf)

        # train the forest
        self.clf.fit(X, y)

        print "Model trained. Feature importances:"
        print(self.clf.feature_importances_)

    def test_command(self):
        # if already tested show the dots and abort
        dots = self.already_tested[self.current_idx]
        if len(dots) > 0:
            # self.show_crosses(dots, self.slider.get())///////////////REPLACE THIS WITH DOT PRINT ITERATOR
            return

        # check if features have been updated
        if not self.updated:
            # update the features
            self.image_feature.update_features(self.imgArray, self.current_idx, True)
            self.updated = True

        if len(self.positive_dataset) == 0:
            self.pos_points = utils.shuffle((np.load('annot_al.npy')[50]), random_state=0)
            self.neg_points = utils.shuffle((np.load('neg_annot_al.npy')[50]), random_state=0)

            for i in range(10):
                p_point = self.pos_points[i]
                n_point = self.neg_points[i]
                feats = self.image_feature.extractFeatsFromPoint(p_point, self.settings.get_selection_mask())
                self.positive_dataset.append(feats)
                # get features from point and append it to negative dataset
                feats = self.image_feature.extractFeatsFromPoint(n_point, self.settings.get_selection_mask())
                self.negative_dataset.append(feats)

            self.train_command()

        # test the frame
        index, dots, self.probabilities, dictionary = test_frame(self.current_idx, self.imgArray,
                                                                 self.image_feature, True, self.settings.get_mser_opts(), self.clf,
                                                                 self.settings.get_selection_mask(),
                                                                 self.settings.get_dots_distance())
        self.image_feature.merge_dictionaries(dictionary)

        # save the dots
        self.already_tested[index] = dots

        # show the crosses
        # self.show_crosses(dots, self.slider.get())

        # update flag
        self.has_been_tested = True

        # start async calls to test function for next and previous frame
        self._test_next_frame()
        self._test_previous_frame()

    def _test_next_frame(self):
        index = self.current_idx + 1

        if index == self.num_frames + 1:
            return

        # check if not been tested before and not retrained
        dots = self.already_tested[index]
        if len(dots) == 0:
            image_array = self.get_image_from_idx(index)
            self.res_plus = self.pool.apply_async(test_frame, args=(index, image_array,
                                                                    self.image_feature, True, self.settings.get_mser_opts(), self.clf,
                                                                    self.settings.get_selection_mask(),
                                                                    self.settings.get_dots_distance()))

    def _test_previous_frame(self):
        index = self.current_idx - 1

        if index == 0:
            return

        # check if not been tested before and not retrained
        dots = self.already_tested[index]
        if len(dots) == 0:
            image_array = self.get_image_from_idx(index)
            self.res_minus = self.pool.apply_async(test_frame, args=(index, image_array,
                                                                     self.image_feature, True, self.settings.get_mser_opts(), self.clf,
                                                                     self.settings.get_selection_mask(),
                                                                     self.settings.get_dots_distance()))

    def slider_command(self, event):
        dots = self.already_tested[self.current_idx]
        # if len(dots) > 0:
            # self.show_crosses(dots, self.slider.get())

    def _return_on_entry(self, event):
        entry_current_text = self.current_idx_entry.get()
        index = 1
        try:
            index = int(entry_current_text)
        except ValueError:
            pass

        if index < 1:
            index = 1
        elif index > self.num_frames:
            index = self.num_frames

        self.current_idx_entry.delete(0, END)
        self.current_idx_entry.insert(END, index)

        self.current_idx = index

        # update the frame
        self.imgArray = self.get_image_from_idx(self.current_idx)
        self.current_image = Image.fromarray(self.select_channel(self.imgArray))
        self.img.paste(self.current_image)

        if self.has_been_tested:
            # get previously calculated results
            if self.res_minus is not None:
                try:
                    idx, dots, self.probabilities, dictionary = self.res_minus.get(0)
                    self.image_feature.merge_dictionaries(dictionary)
                except mp.TimeoutError:
                    pass
                else:
                    self.already_tested[idx] = dots

            if self.res_plus is not None:
                try:
                    idx, dots, self.probabilities, dictionary = self.res_plus.get(0)
                    self.image_feature.merge_dictionaries(dictionary)
                except mp.TimeoutError:
                    pass
                else:
                    self.already_tested[idx] = dots

            self.test_command()

    def _focus_on_entry(self, event):
        self.current_idx_entry.select_range(0, END)

    def _on_left(self, event):
        self.left_command()

    def _on_right(self, event):
        self.right_command()

    def _new_combobox_selection(self, event):
        # update the frame
        self.current_image = Image.fromarray(self.select_channel(self.imgArray))
        self.img.paste(self.current_image)

    def refine_command(self):
        # get previously calculated results, aborts concurrent calls
        if self.res_minus is not None:
            try:
                idx, dots, self.probabilities, dictionary = self.res_minus.get(0)
                self.image_feature.merge_dictionaries(dictionary)
            except mp.TimeoutError:
                pass
            else:
                self.already_tested[idx] = dots

        if self.res_plus is not None:
            try:
                idx, dots, self.probabilities, dictionary = self.res_plus.get(0)
                self.image_feature.merge_dictionaries(dictionary)
            except mp.TimeoutError:
                pass
            else:
                self.already_tested[idx] = dots

        # show the refiner window
        self.refiner.deiconify()

        # create the list of dots along with the index of the frame
        dots_list = list()
        for i, dots in enumerate(self.already_tested):
            if len(dots) > 0:
                dots_list.append((i, dots))

        # start the refiner
        self.refiner.start(dots_list, self.settings.get_low_thresh(), self.settings.get_high_thresh())

    def overlay(self):
        # check if the forest has been tested
        if self.has_been_tested:
            status = self.overlay_button.get()
            if status:
                # display probabilities image
                color_map = plt.get_cmap('jet')
                rgba_probability_img = color_map(self.probabilities, bytes=True)
                probability_image = Image.fromarray(rgba_probability_img)
                self.img.paste(probability_image)
            else:
                self.current_image = Image.fromarray(self.select_channel(self.imgArray))
                self.img.paste(self.current_image)

                dots = self.already_tested[self.current_idx]
                if len(dots) > 0:
                    self.show_crosses(dots, self.slider.get())
        else:
            self.overlay_button.set(0)

    # def show_crosses(self, dots, threshold):
    #     # blank the image
    #     self.current_image = Image.fromarray(self.select_channel(self.imgArray))
    #     self.img.paste(self.current_image)

    #     # iterate over the dots
    #     for dot in dots:
    #         # display only dots with probability larger than threshold
    #         if dot.probability < threshold:
    #             continue

    #         # HERE ARE THE RETURNED POINTS, IN DOT ///////////////////////////////
    #         x, y = dot.x, dot.y
    #         p0, p1, p2, p3 = get_coordinates(x, y, 4)

    #         # display a yellow cross over the dot
    #         draw = ImageDraw.Draw(self.current_image)
    #         draw.line([(p0, p1), (p2, p3)], fill="yellow")
    #         draw.line([(p2, p1), (p0, p3)], fill="yellow")

    #     self.img.paste(self.current_image)

    # method used by Refiner to display current point on the image
    # def display_point(self, point, index):
    #     self.imgArray = self.get_image_from_idx(index)
    #     self.current_image = Image.fromarray(self.imgArray)
    #     self.img.paste(self.current_image)

    #     # updates current index info
    #     self.current_idx = index
    #     self.current_idx_entry.delete(0, END)
    #     self.current_idx_entry.insert(END, self.current_idx)

    #     p0, p1, p2, p3 = get_coordinates(point[1], point[0], self.settings.get_patch_size())
    #     draw = ImageDraw.Draw(self.current_image)
    #     draw.rectangle([(p0, p1), (p2, p3)], outline="white")
    #     self.img.paste(self.current_image)

    # method used by Refiner to update the datasets
    def update_datasets(self, pos_dataset, neg_dataset):
        for point, index_frame in pos_dataset:
            self.image_feature.update_features(None, index_frame, True)
            feats = self.image_feature.extractFeatsFromPoint(point, self.settings.get_selection_mask())
            self.positive_dataset.append(feats)

        for point, index_frame in neg_dataset:
            self.image_feature.update_features(None, index_frame, True)
            feats = self.image_feature.extractFeatsFromPoint(point, self.settings.get_selection_mask())
            self.negative_dataset.append(feats)

        # train again the model
        self.train_command()

        print("Datasets updated.")

    # return the frame corresponding to the specified index

    def get_image_from_idx(self, idx):
        image_path = self.folder_path + "/frame_" + str(idx).zfill(4) + ".png"
        image = misc.imread(image_path)

        return image

    def select_channel(self, image_array):
        copy_img_array = image_array.copy()
        type_cb = self.combobox_value.get()
        if type_cb == 'RGB':
            return copy_img_array
        if type_cb == 'ch1':
            ch = image_array[:, :, 0]
            copy_img_array[:, :, 1] = ch
            copy_img_array[:, :, 2] = ch
            return copy_img_array
        if type_cb == 'ch2':
            ch = image_array[:, :, 1]
            copy_img_array[:, :, 0] = ch
            copy_img_array[:, :, 2] = ch
            return copy_img_array
        if type_cb == 'ch3':
            ch = image_array[:, :, 2]
            copy_img_array[:, :, 0] = ch
            copy_img_array[:, :, 1] = ch
            return copy_img_array

    def get_center_of_mass(self, x, y, offset):
        patchXliminf = max(0, x - offset)
        patchYliminf = max(0, y - offset)
        patchXlimsup = min(512, x + offset)
        patchYlimsup = min(512, y + offset)
        values = self.imgArray[patchYliminf:patchYlimsup, patchXliminf:patchXlimsup, 0]
        cm = ndimage.center_of_mass(values)

        return int(x-offset+cm[1]), int(y-offset+cm[0])

    def track_command(self):
        self.pool.close()

        print '-'*10
        print "Start training, testing and tracking process..."

        num_frames_tracks = self.settings.get_num_frames_tracks()

        image_feature = self.image_feature
        features_mask = self.settings.get_selection_mask()
        print "# Features used: %d" % sum(features_mask)
        print "# Positive annotated points: %d" % len(self.positive_dataset)
        print "# Negative annotated points: %d" % len(self.negative_dataset)

        if len(self.positive_dataset) == 0:
            file_dataset_X = 'X_' + self.folder_path + '.npy'
            file_dataset_y = 'y_' + self.folder_path + '.npy'
            print 'No annotated points.\nLoading the file: ' + file_dataset_X
            pre_loaded_X = np.load(file_dataset_X)
            print 'Loading the file: ' + file_dataset_y
            pre_loaded_y = np.load(file_dataset_y)
            self.clf.fit(pre_loaded_X, pre_loaded_y)
            print 'Model trained. Feature importances:'
            print(self.clf.feature_importances_)
        else:
            self.train_command()

        mser = self.settings.get_mser_opts()
        print "MSER. Delta: %d" % mser[0]
        print "MSER. Min Area: %d" % mser[1]
        print "MSER. Max Area: %d" % mser[2]

        gaps = int(self.settings.gaps_scale.get())

        min_dot_dist = self.settings.get_dots_distance()
        print "Minimum distance between detected points: %d" % min_dot_dist
        confidence = self.slider.get()
        print "Confidence threshold: %.2f" % confidence
        self.root.destroy()
        print '-'*10

        print "NUM FRAMES TO BE TESTED: %d" % num_frames_tracks
        print '-'*10

        start = time.clock()
        print "Start testing frames..."
        detections = list()

        for i in range(1, num_frames_tracks + 1):
            img_array = self.get_image_from_idx(i)
            _, dots, _, _ = test_frame(i, img_array, self.image_feature, False, mser, self.clf, features_mask, min_dot_dist)
            detection_img = list()
            for dot in dots:
                if dot.probability > confidence:
                    detection_img.append((dot.x, dot.y))
            detections.append(detection_img)

        print "End testing frames."
        ts = time.time()
        dts = datetime.datetime.fromtimestamp(ts).strftime('%m%d_%H%M%S')
        np.save('detections_' + dts, detections)
        print "Detections have been saved in file: detections_" + dts + ".npy"

        print '-'*10
        print "Start finding tracks..."
        print "# Max gaps between detections: %d" % gaps

        tracker = Vt.ViterbiTracker(detections)
        track_file_path = "tracks_" + dts + ".txt"

        tracker.start(track_file_path, gaps)

        print time.clock() - start
        plotter = P3D.Plotter3D(track_file_path)

        print "End of the program."
        exit(0)


def test_frame(idx, image_array, image_feature, memory_opt, mser_opts, classifier, features_mask, min_dot_distance):
    image_feature.update_features(image_array, idx, memory_opt)

    # get candidate points in the image from MSER
    red_channel = image_array[:, :, 0]
    red_channel = cv2.equalizeHist(red_channel)  # equalizes the histogram

    mser = cv2.MSER(mser_opts[0], _min_area=mser_opts[1], _max_area=mser_opts[2])
    regions = mser.detect(red_channel)

    candidate_points = set()
    for r in regions:
        for point in r:
            candidate_points.add(tuple(point))

    candidate_points = list(candidate_points)

    X = np.zeros((len(candidate_points), sum(features_mask)))

    # for each candidate point, extract the features and build test matrix
    for i in xrange(len(candidate_points)):
        cp = candidate_points[i]
        feats = image_feature.extractFeatsFromPoint((cp[1], cp[0]), features_mask)
        X[i, :] = feats

    # test the vector
    predictions = classifier.predict_proba(X)

    probabilities = np.zeros((512, 512))

    index = 0
    for i in candidate_points:
        probabilities[i[1], i[0]] = predictions[index][1]
        index += 1

    # dot detector
    dd = Dd.DotDetect(probabilities)
    dots = dd.detect_dots(min_dot_distance)
    del(dd, mser)

    print("Frame " + str(idx) + " has been tested.")

    # return the dots and probabilities image
    # it also returns the updated image feature dictionary
    return idx, dots, probabilities, image_feature.feats_dictionary


def merge_datasets(positive_dataset, negative_dataset):
    positive_dataset = np.array(positive_dataset)
    negative_dataset = np.array(negative_dataset)

    num_pos_samples = positive_dataset.shape[0]
    num_neg_samples = negative_dataset.shape[0]

    X = np.concatenate((positive_dataset, negative_dataset))
    y = np.concatenate((np.ones((num_pos_samples,)), np.zeros((num_neg_samples,))))

    return X, y


def get_patch_coordinates(x, y, offset):
    patchXliminf = max(0, x - offset)
    patchYliminf = max(0, y - offset)
    patchXlimsup = min(512, x + offset)
    patchYlimsup = min(512, y + offset)
    patchXs = range(patchXliminf, patchXlimsup)
    patchYs = range(patchYliminf, patchYlimsup)
    patch = list(itertools.product(patchXs, patchYs))
    return patch


def get_coordinates(x, y, offset):
    return max(0, x - offset), max(0, y - offset), min(512, x + offset), min(512, y + offset)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Error: correct usage is SmartAnnotator sequence_folder number_frames number_channels"
        print "Where sequence_folder is the path of the folder containing the frames"
        exit(-1)

    sa = SmartAnnotator(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    #sa = SmartAnnotator("01corrected", 1209, 2)
    #sa = SmartAnnotator("02corrected", 327, 3)