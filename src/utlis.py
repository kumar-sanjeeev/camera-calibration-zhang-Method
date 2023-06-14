import cv2
import numpy as np
import os
from typing import List
import copy
import argparse
import scipy.optimize

def loadImages( folder_path: str) -> List[np.ndarray]:
    """Reading the images

    Args: 
        folder_path (str): The folder path containing the images

    Returns:
        List [arrays]: Images
    """

    img_files = os.listdir(folder_path)
    print("Loading files from path: ", folder_path, "\n")
    images = []

    for f in img_files:
        img_path = folder_path + "/" + f
        img = cv2.imread(img_path)
        if img is not None and type(img)== np.ndarray:
            images.append(img)
        else:
            print("Error in loading the images")
    
    return images


def drawCorners(img: np.ndarray, corners: List, file_name: str)-> None:
    """Draws circle on the corners
    
    Args:
        img (np.array): The input image
        corners (List): The List of detected corners in the given images
        file_name (str): The output file name
    """

    img = copy.deepcopy(img)
    #draw circles on corners
    for i in range(len(corners)):
        cv2.circle(img=img, 
                   center=(int(corners[i][0]), int(corners[i][1])), 
                   radius=7, 
                   color=(0,0,255), 
                   thickness=-1)
        
    #save in Result dir
    output_dir_path = "Result/original_corners/"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        cv2.imwrite(filename=output_dir_path + file_name + ".png", img=img)
    else:
        cv2.imwrite(filename=output_dir_path + file_name + ".png", img=img)


def getImagesCorners(images: List[np.ndarray], chessBoardSize_xy: tuple) -> np.ndarray:
    """Getting the coordinates of the corners present in the Images
    
    Args:
        images (List): The list of images
        chessBoardSize_xy (tuple): The length and Breadth of printed chessboard image.
    
    Returns:
        Numpy array containing images corners
    """

    all_images_corners = []
    img_no = 1
    
    for img in images:
        ret, corners = cv2.findChessboardCorners(image=img, patternSize=chessBoardSize_xy)
        #if corners detected
        if ret == True:
            img_corners = corners.reshape(-1, 2)
            all_images_corners.append(img_corners)
        drawCorners(img=img, corners=img_corners, file_name="Image"+ str(img_no))
        img_no += 1
    
    return np.array(all_images_corners)



def getWorldCorners(chessBoardSize_xy: tuple, checker_size: float)-> np.ndarray:
    """Getting the 2D coordinates of corners present in printed ChessBoard Pattern
    
    Args:
        chessBoardSize_xy (tuple): The length and Breadth of printed chessboard image.
        checker_size (float): The length of the square checker present in chessboard pattern
    """
    world_corners = []

    """
    c9  #   #  c54
    .   .   .   .
    .   .   .   .
    #   #   #   # 
    #   #   #   #
    c2  #   #   #
    c1  #   #   #
    """
    # along y (horizontal) as per given chessboard axis in given image
    for i in range(1, chessBoardSize_xy[1] + 1):
        # along x (vertical) as per given chessboard axix in given image
        for j in range(1, chessBoardSize_xy[0] + 1):
            world_corners.append((i*checker_size, j*checker_size))
    
    return np.array(world_corners)


def getH(img_corners: np.ndarray, world_corners: np.ndarray)-> np.ndarray:
    """Gets Homography matrix

    Args:
        img_corners (np.ndarray): The corners present in the image
        world_corners(np.ndarray): The corners present in the printed chessboard pattern
    
    Returns:
        Homography [3x3] matrix
    """
    
    h = []
    # no of corners should be same
    if np.shape(img_corners) == np.shape(world_corners):
        for i in range(len(img_corners)):
            x_i = img_corners[i][0]
            y_i = img_corners[i][1]
            X_i = world_corners[i][0]
            Y_i = world_corners[i][1]

            row1 = np.array([-X_i, -Y_i, -1, 0, 0, 0, x_i*X_i, x_i*Y_i, x_i])
            h.append(row1)

            row2 = np.array([0, 0, 0, -X_i, -Y_i, -1, y_i*X_i, y_i*Y_i, y_i])
            h.append(row2)
    
    # apply SVD
    h = np.array(h)
    U, S, V_T = np.linalg.svd(a=h, full_matrices=True)
    V = V_T.T
    H = V[:, -1] # last column: corresponds to smallest eigen value
    H = H/ H[8] # scale ambiguity
    H = np.reshape(a=H, newshape=(3,3))

    return H


def getAllH(images_corners: np.ndarray, world_corners: np.ndarray)-> np.ndarray:
    """Gets Homography of all images present in calibration images dataset
    
    Args:
        img_corners (np.ndarray): The corners present in the image
        world_corners(np.ndarray): The corners present in the printed chessboard pattern
    
    Returns:
        Homography matrices of all images [img_count, 3, 3]
    """

    H_all = []

    for img_corners in images_corners:
        H = getH(img_corners=img_corners, world_corners=world_corners)
        H_all.append(H)
    
    return np.array(H_all)


def getVij(hi: np.ndarray, hj: np.ndarray)-> np.ndarray:
    """
    hi : ith column of matrix H
    hj: jth column of matri H
    """
    Vij = np.array(
        [
            hi[0]* hj[0],
            hi[0]* hj[1] + hi[1]*hj[0],
            hi[1]*hj[1],
            hi[2]*hj[0] + hi[0]*hj[2],
            hi[2]*hj[1] + hi[1]*hj[2],
            hi[2]*hj[2]
        ]
    )

    return Vij.T

def getV(H_all: np.ndarray) -> np.ndarray:
    v = []

    for H in H_all:
        h1 = H[:, 0] # first column of H matrix
        h2 = H[:, 1] # second column of H matrix

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11-v22).T)
    
    return np.array(v) # shape (2*images, 6)


def arrangeB(b):
    B = np.zeros((3,3))
    B[0, 0] = b[0]
    B[0, 1] = b[1]
    B[0, 2] = b[3]
    B[1, 0] = b[1]
    B[1, 1] = b[2]
    B[1, 2] = b[4]
    B[2, 0] = b[3]
    B[2, 1] = b[4]
    B[2, 2] = b[5]

    return B

def getB(H_all):
    v = getV(H_all)
    U, S, V_T = np.linalg.svd(v)
    b = V_T.T[:, -1] # last column
    B = arrangeB(b)

    return B

def getA(B):
    v0 = (B[0,1]* B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)
    lambd = (
        B[2,2] - (B[0,2]**2 + v0* (B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0]
    )
    alpha = np.sqrt(lambd / B[0,0])

    beta = np.sqrt((lambd* B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -1* B[0,1]* (alpha**2)* (beta) / lambd
    u0 = (gamma*v0 /beta) - (B[0,2]* (alpha**2)/lambd)

    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    return A

def getRotAndTrans(A, H_all):
    """Gets the rotation and translation of the each image"""
    RT_all = []

    for H in H_all:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        lambd = 1 / np.linalg.norm(np.matmul(np.linalg.pinv(A), h1), 2)
        r1 = np.matmul(lambd* np.linalg.pinv(A), h1)
        r2 = np.matmul(lambd* np.linalg.pinv(A), h2)
        r3 = np.cross(r1, r2)
        t = np.matmul(lambd*np.linalg.pinv(A),h3)
        RT = np.vstack((r1, r2, r3, t)).T
        RT_all.append(RT)
    
    return RT_all

def extractParamFromA(A: np.ndarray, K_distortion_init: np.ndarray) -> np.ndarray:
    """Extract the individual intrinsics parameters from A matrix"""
    alpha = A[0,0]
    gamma = A[0,1]
    u0 = A[0,2]
    beta = A[1,1]
    v0 = A[1,2]
    k1 = K_distortion_init[0]
    k2 = K_distortion_init[1]

    return np.array([alpha, gamma, beta, u0, v0, k1, k2])

def retrieveA(x0):
    """Get Back A Matrix from the input vector"""
    alpha, gamma, beta, u0, v0, k1, k2 = x0

    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    K_distortion = np.array([k1, k2])

    return A, K_distortion


def reprojectionRMSError(A: np.ndarray, 
                         K_distortion: np.ndarray, 
                         RT_all: np.ndarray, 
                         images_corners : np.ndarray,
                         world_corners: np.ndarray) -> np.ndarray:
    """Calculates the reprojection error of all images"""

    alpha, gamma, beta, u0, v0, k1, k2 = extractParamFromA(A=A, 
                                                           K_distortion_init=K_distortion)
    
    error_all_images = []
    reprojected_corners_all = []

    # images_corners: shape [images, no_corners, 2]
    for i in range(len(images_corners)): # loop over images
        img_corners = images_corners[i]
        RT = RT_all[i] # pose from which img was taken
        P_matrix = np.dot(A, RT) # projection matrix
        error_per_img = 0
        reprojected_img_corners = []

        for j in range(len(img_corners)): # loop over img corners
            world_point_corners_nonHomo_2d = world_corners[j]
            world_point_3d_Homo = np.array(                         # [4,1]
                [
                    [world_point_corners_nonHomo_2d[0]],
                    [world_point_corners_nonHomo_2d[1]],
                    [0],
                    [1],
                ], dtype= float
            )

            img_corner_nonHomo = img_corners[j]
            img_corner_Homo = np.array(                            # [3,1]
                [
                    [img_corner_nonHomo[0]],
                    [img_corner_nonHomo[1]],
                    [1]
                ], dtype=float
            )

            # pixel coordinates (u,v) using Projection Matrix P
            pixel_coords = np.matmul(P_matrix, world_point_3d_Homo)
            u = pixel_coords[0] / pixel_coords[2]
            v = pixel_coords[1]  / pixel_coords[2]


            # image coordinates (or coordinates in camera plane) using only RT matrix
            image_coords = np.matmul(RT, world_point_3d_Homo)
            x_norm = image_coords[0] / image_coords[2]
            y_norm = image_coords[1] / image_coords[2]

            r = np.sqrt(x_norm**2 + y_norm**2)

            u_hat = u + (u-u0)* (k1* r**2 + k2* (r**4))
            v_hat = v + (v-v0)* (k1* r**2 + k2* (r**4))

            img_corner_Homo_hat = np.array(
                [u_hat, 
                 v_hat,
                 [1]], dtype=float
            )

            reprojected_img_corners.append((img_corner_Homo_hat[0],
                                            img_corner_Homo_hat[1]))
            
            error_per_corner = np.linalg.norm(
                (img_corner_Homo - img_corner_Homo_hat), 2
            )
            error_per_img = error_per_img + error_per_corner
        
        reprojected_corners_all.append(reprojected_img_corners)
        error_all_images.append(error_per_img / 54)
    
    return np.array(error_all_images), np.array(reprojected_corners_all)


# loss function required to put as argument in scipy.optimize.leastSquares
def loss_func(x0: np.ndarray,
              RT_all: np.ndarray,
              images_corners: np.ndarray,
              world_corners: np.ndarray) -> np.ndarray:
    A, K_distortion = retrieveA(x0)

    error_all_images, _ = reprojectionRMSError(A=A,
                                               K_distortion=K_distortion,
                                               RT_all=RT_all,
                                               images_corners=images_corners,
                                               world_corners=world_corners)
    
    return np.array(error_all_images)
