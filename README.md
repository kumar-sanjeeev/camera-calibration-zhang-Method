# Zhang's Camera Calibration method Implementation
This project describes the implementation of camera calibration from scratch as described by Zhengyou Zhang in paper ["A Flexible New Technique for Camera Calibration"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf) to estimate camera Intrinsics and distortation parameters.


- **Camera calibration means estimating any camera's intrinsics, extrinsics and distortion parameters. Intrinsic parameters consists of focal length and principal point position and distortion parameters are coefficients of distortion**

$$
A =
\begin{pmatrix}
\alpha & \gamma & u_0\\
0 & \beta & v_0 \\
0 & 0 &1
\end{pmatrix}
$$

$(u_0,v_0)$ is optical center or the principal point in pixel coordinates.

$(\alpha,\beta)$ are scale factors

$\gamma$ is the skewness


# Data
The Zhang’s paper relies on a calibration target (checkerboard in our case) to estimate camera intrinsic parameters. The calibration target used can be found in the file checkerboardPattern.pdf. This was printed on an A4 paper and the size of each square was 21.5mm. Thirteen images taken from a Google Pixel XL phone with focus locked can be accessed from 'CalibrationImgs' folder which we will use to calibrate.

# Procedure

## Initial Parameters Estimation

For camera calibration we need images data. Here, we have 13 images of checkerboard of known size taken from Google Pixel XL smartphone with focal length locked. The actual size of checkerboard square is 21.5 mm. Given checkerboard is of (7x10) size, but we will be neglecting the corner columns and rows, hence final working size is (6x7).

### Approximating camera intrinsic and extrinsic matrix
Steps:
- Find the corners of checkerboard in both world co-ordinates and pixel co-ordinates.
- Compute Homography matrix for each image.
- Use Homography matrices of different images (atleast 3) to formulate the constraint to get the B Matrix.
- Solve the B matrix as described in Appendix B of Zhang's paper to get initial estimate to camera intrinsic parameters.
- Estimate the camera extrinsics parameters by following equations mentioned in paper.

### Approximating Distortion
- We assume minimum camera distortion so we take [0,0] as initial guess for distortion coefficients. We are only considering the first two terms for calculating distortion.
  
## Optimization
Here we minimize the least square distance between actual corner pixel co-ordinates and projected co-ordinates (using estimated parameters). For this we use scipy.optimize.least_squares function.

## Results:

### Initial Approximation

- Intrinsic Parameter Matrix:

    $$
    A_{init} =
    \begin{bmatrix}
    2055.8 & -0.306 & 763.8\\
    0 & 2038.8 & 1348.3 \\
    0 & 0 &1
    \end{bmatrix}
    $$

- Radial Distortion coefficients:

    $$
    K_{init} =
    \begin{bmatrix}
    0 & 0
    \end{bmatrix}
    $$

### After Optimization

- Intrinsic Parameter Matrix:

    $$
    A_{init} =
    \begin{bmatrix}
    2055.8 & -0.306 & 763.8\\
    0 & 2038.8 & 1348.3 \\
    0 & 0 &1
    \end{bmatrix}
    $$

- Radial Distortion coefficients:

    $$
    K_{init} =
    \begin{bmatrix}
    0.0123 & -0.088
    \end{bmatrix}
    $$


# How to run the code
- Go to the root directory
- Run following command
```
python3 src/main.py
```