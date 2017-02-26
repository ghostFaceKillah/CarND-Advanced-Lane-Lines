import cv2
import numpy as np

from utils import show_transformation_on_test_images
from color_grad import combined_thresholding
from undistort import undistort_factory

def make_warpers(w=1280, h=720):
    bottom_w = w
    top_w = 249 
    bottom_h = h - 20
    top_h = bottom_h - 228 
    delta_w = 0 

    region_vertices = np.array([[((w - bottom_w) // 2 + delta_w, bottom_h),
                                 ((w - top_w) // 2 + delta_w, top_h),
                                 ((w + top_w) // 2 + delta_w, top_h),
                                 ((w + bottom_w) // 2 + delta_w, bottom_h)]],
                               dtype=np.float32)

    offsetH = 10
    offsetW = 100
    dest_vertices = np.array([[(offsetW, h - offsetH),
                               (offsetW, offsetH),
                               (w - offsetW, offsetH),
                               (w - offsetW, h - offsetH)]],
                               dtype=np.float32)

    M = cv2.getPerspectiveTransform(region_vertices, dest_vertices)
    Minv = cv2.getPerspectiveTransform(dest_vertices, region_vertices)

    warp = lambda img: cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    unwarp = lambda img: cv2.warpPerspective(img, Minv, (w, h), flags=cv2.INTER_LINEAR)
    return warp, unwarp


def show_warping():
    warp, unwarp = make_warpers()
    show_transformation_on_test_images(warp, 'warp')
    

def show_undistort_warp():
    from undistort import undistort_factory
    undst = undistort_factory()
    warp, unwarp = make_warpers()
    f = lambda img: warp(undst(img))
    show_transformation_on_test_images(f, 'undist_warp')


def show_pipeline():
    undst = undistort_factory()
    warp, unwarp = make_warpers()
    f = lambda img: warp(255 * combined_thresholding(undst(img)))
    show_transformation_on_test_images(f, 'pre_linefind_pipeline')



if __name__ == '__main__':
    # show_warping()
    # show_undistort_warp()
    show_pipeline()
