""" The following functions are the helper functions to run the app."""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt



DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def get_data_path(filename):
    """
    Return the path to a data file.
    """
    return str((DATA_DIR / filename).resolve())

def get_result_path(filename):
    """
    Return the path to a data file.
    """
    return str((RESULTS_DIR / filename).resolve())

def imread(path, flag=cv2.IMREAD_COLOR, rgb=False, normalize=False):
    """
    Read an image from a file.

    path: Image path
    flag: flag passed to cv2.imread
    normalize: normalize the values to [0, 1]
    rgb: convert BGR to RGB
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.imread(str(path), flag)

    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float64) / 255
    return img

def imread_alpha(path, normalize=False):
    """
    Read an image from a file.
    Use this function when the image contains an alpha channel. That channel
    is returned separately.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if normalize:
        img = img.astype(np.float64) / 255

    alpha = img[:,:,-1]
    img = img[:,:,:-1]
        
    return img, alpha


def imshow(img, title=None, flag=cv2.COLOR_BGR2RGB):
    """
    Display the image.
    """
    plt.figure()
    if flag is not None:
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        img = cv2.cvtColor(img, flag)
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.show()


def imwrite(path, img, flag=None):
    """
    Write the image to a file.
    """
    assert type(img) == np.ndarray
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    cv2.imwrite(str(path), img)


def sift_matches(img1, img2):
    """
    Obtain point correspondences using SIFT features.
    img1: First image
    img2: Second image

    src_pts: Nx2 points in img1
    dest_pts: Nx2 corresponding points in img2
    """

    if img1.dtype == np.float64 or img1.dtype == np.float32:
        img1 = (img1 * 255).astype(np.uint8)
    if img2.dtype == np.float64 or img2.dtype == np.float32:
        img2 = (img2 * 255).astype(np.uint8)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])

    src_pts = np.asarray(
        [kp1[good[i][0].queryIdx].pt for i in range(len(good))])
    dest_pts = np.asarray(
        [kp2[good[i][0].trainIdx].pt for i in range(len(good))])

    return src_pts, dest_pts


def click_pts(img, N, title=None):
    """
    Helper function to select points on an image by clicking.
    img: Image to display
    N: Number of points to click

    Return
    pts: Nx2
    """
    fig = plt.figure()
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    else:
        plt.title("Click on point correspondences...")
    pts = np.asarray(plt.ginput(N))
    plt.close(fig)

    return pts

def show_correspondences(src_img, dest_img, src_pts, dest_pts, title=None,
                         show_every_n=1):
    """
    Visualize correspondences between two images by plotting both images
    side-by-side and drawing lines between each point correspondence.

    Since the correspondences may be dense, show_every_n controls how many
    point correspondences to show.

    src_img: source image
    dest_img: destination image
    src_pts: point correspondences in the source image (Nx2)
    dest_pts: point correspondences in the destination image (Nx2)
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2

    N = src_pts.shape[0]

    fig, ax = plt.subplots()
    plt.axis("off")

    padding = 80

    ax.imshow(np.hstack(
        (src_img, np.full((src_img.shape[0], padding, 3), 255, np.uint8), dest_img)))
    t = src_img.shape[1] + padding
    for i in range(0, N, show_every_n):
        # Draw line
        xs = src_pts[i,:]
        xd = dest_pts[i,:]
        ax.plot([xs[0], xd[0]+t], [xs[1], xd[1]], 'r-', linewidth=0.75)

    if title is not None:
        plt.title(title)

    return fig
