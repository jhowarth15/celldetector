__author__ = 'Daniele'

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Plotter3D:
    def __init__(self, track_file_path):
        tracks_file = open(track_file_path, "r")
        tracks = list()
        for line in tracks_file:
            track_arr = np.array(eval(line))
            if len(track_arr > 1):
                tracks.append(track_arr)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for track in tracks:
            ax.plot(track[:, 1], track[:, 2], track[:, 0])
        plt.show()