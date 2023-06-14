import argparse
from utlis import *
from tqdm import tqdm

def ZhangCameraCalib():
    
    parser = argparse.ArgumentParser(
        description="Provide the folder path containing the calibration images and output files path"
    )

    parser.add_argument(
        "--CalibrationImgsPath",
        type=str,
        metavar="",
        default="./calibration_data/calibration_Imgs",
        help="calibration images folder path",
    )

    parser.add_argument(
        "--SaveFolderPath",
        type=str,
        metavar="",
        default="./Result",
        help="Output images Folder Path",
    )

    args = parser.parse_args()

    input_dir_path = args.CalibrationImgsPath
    output_dir_path = args.SaveFolderPath

    images = loadImages(folder_path=input_dir_path)
    chessBoardSize = (9,6) 
    checker_size = 21.5 # mm

    all_images_corners = getImagesCorners(
        images=images,
        chessBoardSize_xy=chessBoardSize
    )

    world_corners = getWorldCorners(chessBoardSize_xy=chessBoardSize,
                                    checker_size=checker_size)
    print("Images corners: ", all_images_corners.shape)
    print("World corners: ", world_corners.shape)

    print(f"\n>>>>>>>>>>>>>> CALCULATING HOMOGRAPHY MATRIX for {len(images)} images <<<<<<<<<<<<<<<<<<<<<<<<")
    H_all = getAllH(images_corners=all_images_corners,
                    world_corners=world_corners)
    B = getB(H_all=H_all)
    print("\n>>>>>>>>>>>>>>> CALCULATING B MATRIX <<<<<<<<<<<<<<<<<<<\n")
    B = getB(H_all)
    print("====== Estimated B matrix is ========: \n", B)
    print("\n>>>>>>>>>>>> CALCULATING A [INTRINSIC PARAMETERS] MATRIX FOR INITIALIZATION <<<<<<<<<<<<<")
    
    A_init = getA(B)
    print("\n===== Initialized A as:======= \n", A_init)
    print("\n>>>>>>>>>>>> CALCULATING ALL IMAGES ROTATION AND TRANSLATION MATRICES [EXTRINSICS PARAMETERS] <<<<<<<<<<<<<\n")
    RT_all = getRotAndTrans(A=A_init, H_all=H_all)
    K_distortion_init = np.array([0,0])
    print("===== Initialize the radial distortation parameters as:=====\n", K_distortion_init, "\n")

    print(">>>>>>>>>>>>> Calculating Initial mean error and reprojection error <<<<<<<<<<<<<<<<<<<<<")

    error_all_images, reprojected_points = reprojectionRMSError(A=A_init,
                                                                K_distortion=K_distortion_init,
                                                                RT_all=RT_all,
                                                                images_corners=all_images_corners,
                                                                world_corners=world_corners)
    print("\nThe mean error before optmization is: ", np.round(np.mean(error_all_images), decimals=3))

 
    print("\n>>>>>>>>>>>>>> RUNNING LEAST SQUARE [OPTIMIZATION] <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    x0 = extractParamFromA(A=A_init, K_distortion_init=K_distortion_init)

    res = scipy.optimize.least_squares(
        fun=loss_func,
        x0=x0,
        method="lm",
        args=[RT_all, all_images_corners, world_corners],
    )

    x1 = res.x
    A_new, K_distortion_new = retrieveA(x1)
    print("\n>>>>>>>>>>>>>> AFTER OPTIMIZATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("\n===== Optimized A matrix [Intrinsics Parameters]:======= \n", A_new)
    print("\n===== Radial distortation parameters after optimization :======= \n", K_distortion_new)

    print("\n>>>>>>>>>>>>> Calculating Initial mean error and reprojection error <<<<<<<<<<<<<<<<<<<<<")

    error_all_images, reprojected_points = reprojectionRMSError(A=A_new,
                                                                K_distortion=K_distortion_new,
                                                                RT_all=RT_all,
                                                                images_corners=all_images_corners,
                                                                world_corners=world_corners)
    
    print("\nThe mean error after optmization is: ", np.round(np.mean(error_all_images), decimals=3))
    

if __name__ == "__main__":
    ZhangCameraCalib()