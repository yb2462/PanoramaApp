import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import src as solutions
import utils


def Panorama():
    """
    Panorama is the "main" interface that lets you execute all the
    functions in homework src.py. It lists a set of
    functions corresponding to the problems that need to be solved in order to have a working PanoramaApp.

    Under the hood the program does the following:
    - compute Homography
    - use RANSAC to improve Homograpgy
    - warp the images
    - Stitch and blend the images
    """
    np.set_printoptions(precision=4, suppress=True)
    # Panorama on your own images.
    panorama = solutions.build_your_own_panorama()

    panorama = np.clip(panorama, 0, 1)
    utils.imshow(panorama, flag=None)
    utils.imwrite(utils.get_result_path("your_picture.png"), panorama,
                  flag=cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    Panorama()
