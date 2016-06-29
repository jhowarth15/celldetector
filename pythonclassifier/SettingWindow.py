__author__ = 'Daniele'


from Tkinter import Checkbutton, Tk, Scale, Spinbox, Label, Entry, Toplevel, IntVar, PhotoImage, PanedWindow, Button, VERTICAL, END, HORIZONTAL
import ttk
import numpy as np
from PIL import Image, ImageTk
import cv2
import random


class SettingWindow(Toplevel):
    def __init__(self, master, max_num_features, num_frames, mser_image):
        Toplevel.__init__(self, master)

        self.protocol('WM_DELETE_WINDOW', self.withdraw)

        self.notebook = ttk.Notebook(self)
        frame_feats = ttk.Frame(self.notebook)
        frame_forest = ttk.Frame(self.notebook)
        frame_mser = ttk.Frame(self.notebook)
        frame_other = ttk.Frame(self.notebook)
        self.notebook.add(frame_feats, text="Features ")
        self.notebook.add(frame_forest, text=" Forest  ")
        self.notebook.add(frame_mser, text=" MSER  ")
        self.notebook.add(frame_other, text=" Other  ")

        self.max_num_feats = max_num_features
        self.selection = None

        self.mser_image = mser_image

        rand_row = random.randint(1, 512-200)
        rand_col = random.randint(1, 512-110)
        self.mser_area = mser_image[rand_row:rand_row+180, rand_col:rand_col+100]

        # read images from icons folder
        self.hf0_img = PhotoImage(file="./icons/hf0.gif")
        self.hf1_img = PhotoImage(file="./icons/hf1.gif")
        self.hf2_img = PhotoImage(file="./icons/hf2.gif")
        self.hf3_img = PhotoImage(file="./icons/hf3.gif")
        self.hf4_img = PhotoImage(file="./icons/hf4.gif")
        self.hf5_img = PhotoImage(file="./icons/hf5.gif")

        self.features_vars = list()
        for i in range(max_num_features):
            self.features_vars.append(IntVar())

        Label(frame_feats, text="Patch size (" + u"\N{GREEK SMALL LETTER PI}" + "):").grid(row=0, column=0, pady=5)
        self.patch_size_spinbox = Spinbox(frame_feats, from_=3, to=30, width=3)
        self.patch_size_spinbox.delete(0, END)
        self.patch_size_spinbox.insert(END, 10)
        self.patch_size_spinbox.grid(row=0, column=1, padx=5)

        f1 = ttk.Labelframe(frame_feats, text='Mean filter')
        f1.grid(row=1, columnspan=2)

        Label(f1, text=u"\N{GREEK SMALL LETTER PI}").grid(row=0, column=0)
        Checkbutton(f1, text="R", variable=self.features_vars[0]).grid(row=0, column=1)
        Checkbutton(f1, text="G", variable=self.features_vars[1]).grid(row=0, column=2)
        Checkbutton(f1, text="B", variable=self.features_vars[2]).grid(row=0, column=3)

        Label(f1, text=u"\N{GREEK SMALL LETTER PI}" + "/2").grid(row=1, column=0)
        Checkbutton(f1, text="R", variable=self.features_vars[3]).grid(row=1, column=1)
        Checkbutton(f1, text="G", variable=self.features_vars[4]).grid(row=1, column=2)
        Checkbutton(f1, text="B", variable=self.features_vars[5]).grid(row=1, column=3)

        f2 = ttk.Labelframe(frame_feats, text="Gaussian filter")
        f2.grid(row=2, columnspan=2)

        Label(f2, text=str(1.0)).grid(row=0, column=0)
        Checkbutton(f2, text="R", variable=self.features_vars[6]).grid(row=0, column=1)
        Checkbutton(f2, text="G", variable=self.features_vars[7]).grid(row=0, column=2)
        Checkbutton(f2, text="B", variable=self.features_vars[8]).grid(row=0, column=3)

        Label(f2, text=str(3.5)).grid(row=1, column=0)
        Checkbutton(f2, text="R", variable=self.features_vars[9]).grid(row=1, column=1)
        Checkbutton(f2, text="G", variable=self.features_vars[10]).grid(row=1, column=2)
        Checkbutton(f2, text="B", variable=self.features_vars[11]).grid(row=1, column=3)

        f3 = ttk.Labelframe(frame_feats, text="Laplacian of gaussian")
        f3.grid(row=3, columnspan=2)

        Label(f3, text=str(2.0)).grid(row=0, column=0)
        Checkbutton(f3, text="R", variable=self.features_vars[12]).grid(row=0, column=1)
        Checkbutton(f3, text="G", variable=self.features_vars[13]).grid(row=0, column=2)
        Checkbutton(f3, text="B", variable=self.features_vars[14]).grid(row=0, column=3)

        Label(f3, text=str(3.5)).grid(row=1, column=0)
        Checkbutton(f3, text="R", variable=self.features_vars[15]).grid(row=1, column=1)
        Checkbutton(f3, text="G", variable=self.features_vars[16]).grid(row=1, column=2)
        Checkbutton(f3, text="B", variable=self.features_vars[17]).grid(row=1, column=3)

        f4 = ttk.Labelframe(frame_feats, text="Haar-like features")
        f4.grid(row=1, rowspan=2, column=3, padx=5)

        Checkbutton(f4, image=self.hf0_img, variable=self.features_vars[18]).grid(row=0, column=0)
        Checkbutton(f4, image=self.hf1_img, variable=self.features_vars[19]).grid(row=0, column=1)
        Checkbutton(f4, image=self.hf2_img, variable=self.features_vars[20]).grid(row=1, column=0)
        Checkbutton(f4, image=self.hf3_img, variable=self.features_vars[21]).grid(row=1, column=1)
        Checkbutton(f4, image=self.hf4_img, variable=self.features_vars[22]).grid(row=2, column=0)
        Checkbutton(f4, image=self.hf5_img, variable=self.features_vars[23]).grid(row=2, column=1)

        buttons_paned_window = PanedWindow(frame_feats, orient=VERTICAL)
        buttons_paned_window.grid(row=3, column=3)

        self.select_all_button = Button(buttons_paned_window, text="Select all", command=self._select_all)
        buttons_paned_window.add(self.select_all_button)

        self.clear_selection_button = Button(buttons_paned_window, text="Clear selection", command=self._clear_selection)
        buttons_paned_window.add(self.clear_selection_button)

        # default values
        for j in [0, 1, 3, 6, 7, 9, 15, 21, 23]:
            self.features_vars[j].set(1)

        # FOREST FRAMES
        # number of trees
        f5 = ttk.Labelframe(frame_forest, text="Number of trees")
        f5.grid(row=0, columnspan=2, pady=5, padx=5)
        Label(f5, text="N").grid(row=1, column=0)
        self.num_trees_scale = Scale(f5, from_=5, to=500, resolution=5, orient=HORIZONTAL)
        self.num_trees_scale.set(300)
        self.num_trees_scale.grid(row=0, column=1, rowspan=2)

        # depth single tree
        f6 = ttk.Labelframe(frame_forest, text="Depth single tree")
        f6.grid(row=1, columnspan=2, pady=5, padx=5)
        Label(f6, text="d").grid(row=1, column=0)
        self.depth_tree_scale = Scale(f6, from_=2, to=20, orient=HORIZONTAL)
        self.depth_tree_scale.set(3)
        self.depth_tree_scale.grid(row=0, column=1, rowspan=2)

        # percentage number of features
        f7 = ttk.Labelframe(frame_forest, text="% subset of features")
        f7.grid(row=2, columnspan=2, pady=5, padx=5)
        Label(f7, text="m").grid(row=1, column=0)
        self.percentage_feats_scale = Scale(f7, from_=0.0, to=1, resolution=0.05, orient=HORIZONTAL)
        self.percentage_feats_scale.set(0.5)
        self.percentage_feats_scale.grid(row=0, column=1, rowspan=2)

        # mser frame
        # delta
        f8 = ttk.Labelframe(frame_mser, text="Delta")
        f8.grid(row=0, columnspan=2, pady=5, padx=5)
        Label(f8, text=u"\N{GREEK SMALL LETTER DELTA}").grid(row=1, column=0)
        self.delta_scale = Scale(f8, from_=1, to=10, resolution=1, orient=HORIZONTAL)
        self.delta_scale.set(2)
        self.delta_scale.grid(row=0, column=1, rowspan=2)

        # min area
        f9 = ttk.Labelframe(frame_mser, text="Minimum area")
        f9.grid(row=1, columnspan=2, pady=5, padx=5)
        Label(f9, text="m").grid(row=1, column=0)
        self.min_area_scale = Scale(f9, from_=2, to=200, orient=HORIZONTAL)
        self.min_area_scale.set(10)
        self.min_area_scale.grid(row=0, column=1, rowspan=2)

        # percentage number of features
        f10 = ttk.Labelframe(frame_mser, text="Maximum area")
        f10.grid(row=2, columnspan=2, pady=5, padx=5)
        Label(f10, text="M").grid(row=1, column=0)
        self.max_area_scale = Scale(f10, from_=50, to=1000, resolution=5, orient=HORIZONTAL)
        self.max_area_scale.set(350)
        self.max_area_scale.grid(row=0, column=1, rowspan=2)

        # mser image
        f11 = ttk.Labelframe(frame_mser)
        f11.grid(row=0, rowspan=3, column=2, padx=5)

        self.mser_img_array = Image.fromarray(self.mser_area, "RGB")
        self.mser_img = ImageTk.PhotoImage(self.mser_img_array)

        img_label = Label(f11, image=self.mser_img)
        img_label.grid(row=0, column=0)

        buttons_p_w_mser = PanedWindow(f11, orient=HORIZONTAL)
        try_button = Button(f11, text="Try", command=self.try_mser)
        buttons_p_w_mser.add(try_button)
        change_button = Button(f11, text="New img", command=self.change_mser)
        buttons_p_w_mser.add(change_button)
        buttons_p_w_mser.grid(row=1, column=0)

        # other frame
        f12 = ttk.Labelframe(frame_other, text="Refinement")
        f12.grid(row=0, columnspan=2, pady=5, padx=5)
        Label(f12, text=u"\N{GREEK CAPITAL LETTER PHI}_l").grid(row=1, column=0)
        self.low_thresh_scale = Scale(f12, from_=0, to=1, resolution=0.05, orient=HORIZONTAL, length=90)
        self.low_thresh_scale.set(0.45)
        self.low_thresh_scale.grid(row=0, column=1, rowspan=2)
        Label(f12, text=u"\N{GREEK CAPITAL LETTER PHI}_h").grid(row=3, column=0)
        self.high_thresh_scale = Scale(f12, from_=0, to=1, resolution=0.05, orient=HORIZONTAL, length=90)
        self.high_thresh_scale.set(0.65)
        self.high_thresh_scale.grid(row=2, column=1, rowspan=2)

        f13 = ttk.Labelframe(frame_other, text="Dots distance")
        f13.grid(row=1, columnspan=2, pady=5, padx=5)
        Label(f13, text=u"     \N{GREEK SMALL LETTER SIGMA}").grid(row=1, column=0)
        self.dots_distance_scale = Scale(f13, from_=1, to=20, resolution=1, orient=HORIZONTAL, length=90)
        self.dots_distance_scale.set(6)
        self.dots_distance_scale.grid(row=0, column=1, rowspan=2)

        f14 = ttk.Labelframe(frame_other, text="Tracks")
        f14.grid(row=0, column=3, pady=5, padx=5)
        Label(f14, text="N").grid(row=1, column=0)
        self.num_frames_tracks_spinbox = Spinbox(f14, from_=2, to=num_frames, width=10)
        self.num_frames_tracks_spinbox.delete(0, END)
        self.num_frames_tracks_spinbox.insert(END, num_frames)
        self.num_frames_tracks_spinbox.grid(row=0, column=1, rowspan=2)

        Label(f14, text=u"\N{GREEK SMALL LETTER TAU}").grid(row=3, column=0)
        self.gaps_scale = Scale(f14, from_=1, to=10, resolution=1, orient=HORIZONTAL, length=90)
        self.gaps_scale.set(2)
        self.gaps_scale.grid(row=2, column=1, rowspan=2)

        self.notebook.pack(padx=1, pady=1)

        save_button = Button(self, text=" Save and Close window ", command=self.withdraw)
        save_button.pack(pady=2)

    def _select_all(self):
        for i, var in enumerate(self.features_vars):
            var.set(1)

    def _clear_selection(self):
        for i, var in enumerate(self.features_vars):
            var.set(0)

    def change_mser(self):
        rand_row = random.randint(1, 512-200)
        rand_col = random.randint(1, 512-110)
        self.mser_area = self.mser_image[rand_row:rand_row+180, rand_col:rand_col+100]

        self.update_mser_image(self.mser_area)

    def try_mser(self):
        delta = self.delta_scale.get()
        min_area = self.min_area_scale.get()
        max_area = self.max_area_scale.get()

        image = self.mser_area
        red_c = image[:,:,0]
        red_c = cv2.equalizeHist(red_c)

        det_img = image.copy()

        mser = cv2.MSER(delta, _min_area=min_area, _max_area=max_area)
        regions = mser.detect(red_c)
        cp = list()
        new_c = np.zeros(self.mser_area.shape, dtype=np.uint8)
        for r in regions:
            for point in r:
                cp.append(point)
                det_img[point[1], point[0], 0] = 0
                det_img[point[1], point[0], 1] = 0
                det_img[point[1], point[0], 2] = 204
                #new_c[point[1], point[0]] = 255

        self.update_mser_image(det_img)

    def update_mser_image(self, new_image):
        self.mser_img_array = Image.fromarray(new_image)
        self.mser_img.paste(self.mser_img_array)

    def get_patch_size(self):
        patch_size = self.patch_size_spinbox.get()
        return int(patch_size)

    def get_num_frames_tracks(self):
        num_frames_tracks = self.num_frames_tracks_spinbox.get()
        return int(num_frames_tracks)

    def get_mser_opts(self):
        return [self.delta_scale.get(), self.min_area_scale.get(), self.max_area_scale.get()]

    def get_forest_opts(self):
        return [self.num_trees_scale.get(), self.depth_tree_scale.get(), self.percentage_feats_scale.get()]

    def get_low_thresh(self):
        return self.low_thresh_scale.get()

    def get_high_thresh(self):
        return self.high_thresh_scale.get()

    def get_dots_distance(self):
        return int(self.dots_distance_scale.get())

    def get_selection_mask(self):
        if self.selection is not None:
            return self.selection

        selection_mask = np.zeros((self.max_num_feats, ), dtype='bool')
        for i, var in enumerate(self.features_vars):
            selection_mask[i] = var.get()
        self.selection = selection_mask
        return selection_mask

if __name__ == '__main__':
    root = Tk()
    mser = np.zeros((512,512))
    sw = SettingWindow(root, 24, 1024, mser)
    root.withdraw()
    root.mainloop()
    #sa = SmartAnnotator("02corrected", 327, 3)

