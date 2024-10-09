from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
import sys


import utils


# Tunable parameters

# RANSAC parameters for challenge1a
CHALLENGE_1C_RANSAC_N = 500 # Number of iterations
CHALLENGE_1C_RANSAC_EPS = 5 # Maximum reprojection error




#--------------------------------------------------------------------------
# Challenge 1: Image Mosaicking App
#--------------------------------------------------------------------------
#DONE
def compute_homography(src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Compute the homography matrix relating the given points.
    Hint: use np.linalg.eig to compute the eigenvalues of a matrix.

    Args:
        src_pts (np.ndarray): Nx2 matrix of source points
        dest_pts (np.ndarray): Nx2 matrix of destination points

    Returns:
        np.ndarray: 3x3 homography matrix
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2
    N = src_pts.shape[0]
    A = np.zeros((N*2,9))
    for row in range (N*2):
        if(row%2==0):
            A[row, 0 ] = src_pts[int(row/2), 0]
            A[row, 1] = src_pts[int(row/2),1]
            A[row, 2] = 1
            A[row,6] = -1*dest_pts[int(row/2),0]*src_pts[int(row/2),0]
            A[row, 7] = -1*dest_pts[int(row/2),0]*src_pts[int(row/2),1]
            A[row,8] = -1*dest_pts[int(row/2),0] 
        else:
            A[row, 3 ] = src_pts[int(row/2), 0]
            A[row, 4] = src_pts[int(row/2),1]
            A[row, 5] = 1
            A[row,6] = dest_pts[int(row/2),1]*src_pts[int(row/2),0]*-1
            A[row, 7] = dest_pts[int(row/2),1]*src_pts[int(row/2),1]*-1
            A[row,8] = dest_pts[int(row/2),1] *-1 
    L, X = np.linalg.eig(np.dot(A.T,A)) 
    ind_eig = np.argmin(L)
    homography = X[:,ind_eig]
    homography = homography.reshape(3,3)
    #normalize H
    homography/=homography[-1,-1]
    return homography.reshape(3,3)

#DONE
def apply_homography(H: np.ndarray, test_pts: np.ndarray) -> np.ndarray:
    """
    Apply the homography to the given test points

    Args:
        H (np.ndarray): 3x3 homography matrix
        test_pts (np.ndarray): Nx2 test points

    Returns:
        np.ndarray: Nx2 points after applying the homography
    """
    assert test_pts.shape[1] == 2
    N = test_pts.shape[0]
    z=np.ones((N,1))
    d3_test = np.hstack((test_pts,z))
    d3_test =d3_test.T
    dest_pts= np.dot(H,d3_test)
    dest_pts = dest_pts.T
    #convert from homogeneous to heterogeneous coordinates
    """
    for row in dest_pts:
        row[0] = row[0]/row[2]
        row[1] = row[1]/row[2]
    """
    dest_pts[:, 0] /= dest_pts[:, 2]
    dest_pts[:, 1] /= dest_pts[:, 2]
    return dest_pts[:,:2]

#DONE
def backward_warp_img(
    src_img: np.ndarray, H_inv: np.ndarray, 
    dest_canvas_width_height: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a homography to the image using backward warping.

    Use cv2.remap to linearly interpolate the warped points.
    The function call should follow this form:
        img_warp = cv2.remap(img, map_x.astype(np.float32), 
            map_y.astype(np.float32), cv2.INTER_LINEAR, borderValue=np.nan).
    This casts map_x and map_y to 32-bit floats, chooses linear interpolation,
    and sets pixels outside the original image to NaN (not-a-number).
        
    Also, since we are working with color images, you should process each
    color channel separately.

    Args:
        src_img (np.ndarray): Nx2 source points
        H_inv (np.ndarray): 3x3 inverse of the src -> dest homography
        dest_canvas_width_height (Tuple[int, int]): size of the destination image

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        binary mask where the destination image is filled in, final image
    """

    d_width, d_height = dest_canvas_width_height
    X,Y = np.meshgrid(np.arange(d_width),np.arange(d_height))
    #Apply the inverse homography can use apply_homography?
    dest_coords = np.column_stack((X.flatten(),Y.flatten()))
    #reshape
    src_coords = apply_homography(H_inv,dest_coords)
    map_x = src_coords[:,0].reshape((d_height,d_width))
    map_y = src_coords[:,1].reshape((d_height,d_width))
    final_img=cv2.remap(src_img, map_x.astype(np.float32),map_y.astype(np.float32),cv2.INTER_LINEAR, borderValue=np.nan)
    #Display the image using matplotlib
    plt.imshow(final_img)
    plt.show()
    mask = np.isnan(final_img)
    final_img[mask] = 0
    mask = ~mask
    #account for different channels
    for channel in range(3):  # Iterate over R, G, B channels (0, 1, 2)
       mask[:, :, channel] = mask[:,:,0]
    
    return mask, final_img

#DONE
def warp_img_onto(src_img: np.ndarray, dest_img: np.ndarray, 
                  src_pts: np.ndarray, dest_pts: np.ndarray) -> np.ndarray:
    """
    Warp the source image on the destination image.
    Return the resulting image.
    
    Step 1: estimate the homography mapping src_pts to dest_pts
    Step 2: warp src_img onto dest_img using backward_warp_img(..)
    Step 3: overlay the warped src_img onto dest_img
    Step 4: return the resulting image

    Args:
        src_img (np.ndarray): source image
        dest_img (np.ndarray): destination image
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points

    Returns:
        np.ndarray: resulting image with the source image warped on the 
        destination image
    """
    #Step 1: estimate the homography mapping src_pts to dest_pts
    H = compute_homography(src_pts, dest_pts)
    H_inv=np.linalg.inv(H)
    dest_w_h = np.array([dest_img.shape[1],dest_img.shape[0]])

    #Step 2: warp src_img onto dest_img using backward_warp_img(..)
    mask, final_img = backward_warp_img(src_img, H_inv,dest_w_h)

    #Step 3  overlay the warped src_img onto dest_img
    
    res_image = dest_img.copy()  # Make a copy of dest_img to preserve the original
    res_image[mask] = final_img[mask]
    #step 4
    return res_image


#DONE
def run_RANSAC(src_pts: np.ndarray, dest_pts: np.ndarray, ransac_n: int, 
               ransac_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run RANSAC on the given point correspondences to compute a homography 
    from source to destination points.

    Args:
        src_pts (np.ndarray): Nx2 source points
        dest_pts (np.ndarray): Nx2 destination points
        ransac_n (int): number of RANSAC iterations
        ransac_eps (float): maximum 2D reprojection error for inliers

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the inlier indices and the 
        estimated homography matrix.
    """
    assert src_pts.shape[0] == dest_pts.shape[0]
    assert src_pts.shape[1] == 2 and dest_pts.shape[1] == 2
    ran_ind = np.zeros((4))
    src_h = np.zeros((4,2))
    dest_h = np.zeros((4,2))
    h_list = []
    N = src_pts.shape[0]

    #Ransac alg
    for i in range (ransac_n):
        #randomly choose 4 different indices for the points
        ind = 0
        indx = np.random.choice(N,4, False)
        #populate a src_h and dest_h arrays with these random pts
        for element in indx:
            src_h[ind,:] = src_pts[element, :]
            dest_h[ind, :] = dest_pts[element, :]
            ind=ind+1
        
        h = compute_homography(src_h, dest_h)
        M = 0
        in_ind = []
        k = 0
        check = apply_homography(h,src_pts)
        check.astype(float)

        #compute number M of pts that fit the model
        for row in range(check.shape[0]):
            check_pt = check[row,:]
            dest_pt = dest_pts[row,:]
            #calculate the error
            error = np.sqrt((dest_pt[0]-check_pt[0])**2+(dest_pt[1]-check_pt[1])**2)
            if(error<=ransac_eps):
                M+=1
                in_ind.append(k)
            k=k+1
        h_list.append((M,h,in_ind))
    #choose h with the largest M
    largest_M_tuple = max(h_list, key=lambda x: x[0])

    #extract in_ind and est_h correspondent to the max M
    in_ind = largest_M_tuple[2]
    est_h = largest_M_tuple[1]

    #optional - recompute homography from in_ind
    """
    for ind, element in in_ind:
            src_h [ ind] = src_pts[element]
            dest_h[ind] = dest_pts[element]
    h_new = compute_homography(src_h, dest_h)
    """
    return in_ind, est_h

#DONE
def blend_image_pair(img1: np.ndarray, mask1: np.ndarray, img2: np.ndarray, 
                     mask2: np.ndarray, blending_mode: str) -> np.ndarray:
    """
    Blend two images together using the "overlay" or "blend" mode.
    
    In the "overlay" mode, image 2 is overlaid on top of image 1 wherever
    mask2 is non-zero.
    In "blend" mode, the blended image is a weighted combination of the two
    images, where each pixel is weighted based on its distance to the edge.

    Args:
        img1 (np.ndarray): First image
        mask1 (np.ndarray): Mask where the first image is non-zero
        img2 (np.ndarray): Second image
        mask2 (np.ndarray): Mask where the second image is non-zero
        blending_mode (str): "overlay" or "blend"

    Returns:
        np.ndarray: blended image.
    """
    assert blending_mode in ["overlay", "blend"]

    blend_im=img1.copy()
    if blending_mode=="overlay":
        for channel in range(3):
            mask2=mask2.astype(int)
            mask1=mask1.astype(int)
            
            #blend_im[:,:,channel]=(mask1*img1[:,:,channel])
            blend_im[mask2 != 0] = img2[mask2 != 0]

    if blending_mode == "blend":
        #perform weighted blend
        # Compute the blending weights using distance transform
        mask2=mask2.astype(float)
        mask1=mask1.astype(float)

        weight1 = distance_transform_edt(mask1)
        weight2 = distance_transform_edt(mask2)

        #make sure to go through all channels       
        for channel in range(3):
            blend_im[:,:,channel]=(weight1*img1[:,:,channel]+weight2*img2[:,:,channel])/(weight1+weight2+1e-9)
        
    return blend_im

def stitch_imgs(imgs: List[np.ndarray]) -> np.ndarray:
    """
    Stitch a list of images together into an image mosaic.
    imgs: list of images to be stitched together. You may assume the order
    the images appear in the list is the order in which they should be stitched
    together.

    Args:
        imgs (List[np.ndarray]): list of images to be stitched together. You may
        assume the order the images appear in the list is the order in which
        they should be stitched together.

    Returns:
        np.ndarray: the final, stitched image
    """
    assert len(imgs)>0

    img1 = imgs[0]
    i=0

    for img2 in imgs[1:]:
        i=i+1
        #first find the correspondent pts
        src_pts, dest_pts = utils.sift_matches(img2, img1)
        #compute homography
        ind,h = run_RANSAC(src_pts,dest_pts, CHALLENGE_1C_RANSAC_N, CHALLENGE_1C_RANSAC_EPS)

        #find corners of img2
        minx_miny = [0,0]
        minx_maxy=[0,img2.shape[0]]
        maxx_miny=[img2.shape[1],0]
        maxx_maxy = [img2.shape[1],img2.shape[0]]
        corners_orig = np.array([minx_miny,maxx_miny,maxx_maxy,minx_maxy])

        #apply homography to the corners 
        corners_new = apply_homography(h,corners_orig)
        minx = np.min(corners_new[:,0])
        maxx = np.max(corners_new[:,0])
        miny = np.min(corners_new[:,1])
        maxy = np.max(corners_new[:,1])

        #compute the bounding boxcheck where are the coordinates of the new_image
        if(minx<0):
            pad_left = int(np.ceil(0-minx))
        else:
            pad_left = int(0)
        if(maxx>img1.shape[1]):
            pad_right=int(np.ceil(maxx-img1.shape[1]))
        else:
            pad_right = int(0)
        if(miny<0):
            pad_top = int(np.ceil(0-miny))
        else:
            pad_top = int(0)
        if(maxy>img1.shape[0]):
            pad_bottom = int(np.ceil(maxy-img1.shape[0]))
        else:
            pad_bottom = int(0)

        pad_rows = (pad_top,pad_bottom)
        pad_cols = (pad_left,pad_right,)
        img1 = np.pad(img1, (pad_rows, pad_cols,(0,0)))
        
        #get mask1
        if(i==1):
            mask1 = np.zeros((img1.shape[0],img1.shape[1]))
            if(pad_bottom!=0 and pad_right!=0):
                mask1[pad_top:-pad_bottom,pad_left:-pad_right] = 1
            elif(pad_bottom==0 and pad_right==0):
                mask1[pad_top:,pad_left:] = 1
            elif(pad_bottom==0):
                mask1[pad_top:,pad_left:-pad_right] = 1
            elif(pad_right==0):
                mask1[pad_top:-pad_bottom,pad_left:] = 1
        else:
            mask1 = np.pad(mask1, (pad_rows, pad_cols))

        #recompute the correspondent pts
        src_pts_n, dest_pts_n = utils.sift_matches(img2, img1)

        #recompute homography
        ind,h = run_RANSAC(src_pts_n,dest_pts_n, CHALLENGE_1C_RANSAC_N, CHALLENGE_1C_RANSAC_EPS)
        h_inv = np.linalg.inv(h)


        #apply backward warp by warp_onto
        mask2,img2 = backward_warp_img(img2,h_inv,(img1.shape[1],img1.shape[0]))
        img1[mask2] = img2[mask2]

        #apply blending technique
        mask2=mask2.astype(float)
        img1 = blend_image_pair(img1, mask1, img2,mask2[:,:,0], "blend")        

        #comnine the masks
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        mask1 = mask1|mask2[:,:,0]
    return img1


def build_your_own_panorama() -> np.ndarray:
    """
    Build your own panorama using images in your path.
    """
    input_path = utils.get_result_path("/your_path/to_images") 

    # Load images
    # list your file names in the order they should be stitched
    file_names = ["1.jpg", "2.jpg", "3.jpg"]
    imgs = []
    for f_name in file_names:
        img_path = str((Path(input_path) / f_name).resolve())
        img = utils.imread(img_path, flag=None, rgb=True, normalize=True)
        imgs.append(img)
    # Define the target resolution (e.g., 1280x720)
    target_width = 640
    target_height = 480

    for img in imgs:
        # U downsample the image
        img = cv2.resize(img, (target_width,target_height))
    panorama = stitch_imgs(imgs)
    return panorama
