"""
Dealing with camera distortion and calibration.
"""

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import shelve
import tqdm

from utils import full_fname_to_core_name, show_transformation_on_test_images

NX = 9 
NY = 6 


def compute_corners(img, nx=NX, ny=NY, draw_corners_name=None):
    """ Compute corners in the calibration image """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw and display the corners
    if ret and draw_corners_name:
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imsave('out/{}.png'.format(draw_corners_name), img)
    return ret, corners


def illustrate_found_corners():
    for img_fname in glob.glob('camera_cal/*'):
        img = cv2.imread(img_fname)
        draw_corners_fname = 'drawn_corners_' + img_fname.split('/')[-1].split('.')[0]
        compute_corners(img, draw_corners_name=draw_corners_fname)


def compute_undistortion_parameters(nx=NX, ny=NY):
    # 3d points in real world, like [0, 0, 0], [1, 0, 0]
    object_points = np.zeros((ny*nx, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    obj_pt_acc = []
    img_pt_acc = []

    for img_fname in glob.glob('camera_cal/*'):
        img = cv2.imread(img_fname)
        ret, corners = compute_corners(img)

        if ret:
            obj_pt_acc.append(object_points)
            img_pt_acc.append(corners)

    img_size = img.shape[1], img.shape[0]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pt_acc, img_pt_acc,
                                                       img_size, None, None)

    assert ret, "Failed to compute undistortion parameters"

    return {
      "camera matrix": mtx,
      "distortion coefficients": dist,
      "rotation vectors": rvecs,
      "translations vectors": tvecs
    }


def easy_undistortion_params():
    s = shelve.open('stuff.db', writeback=True)
    if 'dist params' in s:
        params = s['dist params']
    else:
        params = compute_undistortion_parameters()
        s['dist params'] = params
    s.close()
    return params['camera matrix'], params['distortion coefficients']


def undistort_factory():
    camera_matrix, distortion_coeffs = easy_undistortion_params()
    f = lambda img: cv2.undistort(img, camera_matrix, distortion_coeffs,
                                  None, camera_matrix)
    return f


def show_undistortion_on_checkboard_images():
    # Perhaps make a one combined plot ?
    undst_f = undistort_factory()

    for img_fname in tqdm.tqdm(glob.glob('camera_cal/*')):
        img = cv2.imread(img_fname)
        fname = full_fname_to_core_name(img_fname)
        undist = undst_f(img)
        cv2.imwrite('out/undist_{}.png'.format(fname), undist)


def show_distortion_on_test_images():
    undist = undistort_factory()
    show_transformation_on_test_images(undist, 'undist')


def writeup_visualisation():
    undst_f = undistort_factory()

    img_fname = 'camera_cal/calibration3.jpg'
    img = cv2.imread(img_fname)
    undist = undst_f(img)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    f.tight_layout()
    ax1.set_title('One of calibration images') # , fontsize=10)
    ax1.imshow(img)
    
    ax2.set_title('Distortion corrected calibration image')
    ax2.imshow(undist) # , cmap='gray')

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig('out/calibration_checkerboard.jpg')
    plt.close()


if __name__ == '__main__':
    # easy_undistortion_params()
    # show_undistortion_on_checkboard_images()
    # show_distortion_on_test_images()
    writeup_visualisation()
