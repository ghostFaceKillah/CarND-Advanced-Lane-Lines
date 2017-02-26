import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import show_transformation_on_test_images


SETTINGS = {
    "kernel_size" : 31,
    "x_abs" : (55, 150),
    "quad" : (50, 255),
    "dir" : (0.70, 1.15),
    "s" : (170, 255),  # s from HLS
    "red" : (200, 255)
}


def sobel_norm_thresh(img, norm, kernel_size, t_min, t_max):
    """
    Detect where abs of sobel operator applied to image is in given range.
    Output is a binary image with 1 where it is the case.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if norm == 'y_abs':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
        norm = np.absolute(sobel)
    elif norm == 'x_abs':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        norm = np.absolute(sobel)
    elif norm == 'quadratic':
        x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, kernel_size)
        y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, kernel_size)
        norm = np.sqrt(x * x + y * y)
    else:
        raise Exception("Unhandled sobel operator norm.")

    # Scale to 8-bit (0 - 255) and convert to np.uint8
    sobel_scaled = np.uint8(255 * norm / np.max(norm))
    mask = np.zeros_like(sobel_scaled)
    mask[(sobel_scaled >= t_min) & (sobel_scaled <= t_max) ] = 1
    return mask


def grad_dir_threshold(img, kernel_size, t_min, t_max):
    """
    On binary output image mark places where in the input image the direction of
    sobel operator is in the input range.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
    abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size))

    # Calculate the direction of the gradient
    direction = np.arctan2(abs_y, abs_x)
    mask = np.uint8(np.zeros_like(direction))

    mask = np.zeros_like(direction, dtype=np.uint8)
    mask[(direction >= t_min) & (direction <= t_max)] = 1
    return mask


def hls_threshold(img, t_min=0, t_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_chan = hls[:,:,2]
    mask = np.zeros_like(s_chan)
    mask[(s_chan > t_min) & (s_chan <= t_max)] = 1
    return mask


def red_threshold(img, t_min, t_max):
    red = img[:,:,0]
    mask = np.zeros_like(red, dtype=np.uint8)
    mask[(red >= t_min) & (red <= t_max)] = 1
    return mask


def combined_thresholding(img, settings=SETTINGS):

    abs_x = sobel_norm_thresh(img, norm='x_abs', 
                              kernel_size=settings['kernel_size'], 
                              t_min=settings['x_abs'][0], 
                              t_max=settings['x_abs'][1])

    quad = sobel_norm_thresh(img, norm='quadratic',
                              kernel_size=settings['kernel_size'], 
                              t_min=settings['quad'][0], 
                              t_max=settings['quad'][1])

    dir_mask = grad_dir_threshold(img, 
                                  kernel_size=settings['kernel_size'], 
                                  t_min=settings['dir'][0], 
                                  t_max=settings['dir'][1])

    hls = hls_threshold(img, t_min=settings['s'][0], t_max=settings['s'][1])
    red = red_threshold(img, t_min=settings['red'][0], t_max=settings['red'][1])

    # Combine gradient thresholds
    comb_bin = np.zeros_like(abs_x)
    comb_bin[(abs_x == 1) | ((dir_mask == 1) & (quad == 1))] = 1

    # Combine previous + Color S + Color R
    combined = np.zeros_like(comb_bin)
    combined[(comb_bin == 1) | (hls == 1) | (red == 1)] = 1

    return combined


def older_visualization_code():
    # TODO: remove
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)
    
    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def showcase_sobel_norm_thresholding():
    f = lambda img: 255 * sobel_norm_thresh(img, norm='x_abs', 
                              kernel_size=SETTINGS['kernel_size'], 
                              t_min=SETTINGS['x_abs'][0], 
                              t_max=SETTINGS['x_abs'][1])

    show_transformation_on_test_images(f, 'abs_x_thresholding')


def showcase_quadratic_norm_thresholding():
    quad = lambda img: 255 * sobel_norm_thresh(img, norm='quadratic',
                                  kernel_size=SETTINGS['kernel_size'], 
                                  t_min=SETTINGS['quad'][0], 
                                  t_max=SETTINGS['quad'][1])

    show_transformation_on_test_images(quad, 'quad_thresh')


def showcase_directional_thresholding():
    dir_masking = lambda img: 255 *  grad_dir_threshold(img, 
                                  kernel_size=SETTINGS['kernel_size'], 
                                  t_min=SETTINGS['dir'][0], 
                                  t_max=SETTINGS['dir'][1])
    show_transformation_on_test_images(dir_masking, 'directional')


def showcase_hls_thresholding():
    hls = lambda img: 255 * hls_threshold(img, t_min=SETTINGS['s'][0],
                                         t_max=SETTINGS['s'][1])
    show_transformation_on_test_images(hls, 'hls')


def showcase_red_channel_thresholding():
    red = lambda img: 255 * red_threshold(img, t_min=SETTINGS['red'][0],
                                         t_max=SETTINGS['red'][1])
    show_transformation_on_test_images(red, 'red')


def showcase_combined_thresholding():
    f = lambda img: 255 * combined_thresholding(img)
    show_transformation_on_test_images(f, 'pipeline')


if __name__ == '__main__':
    # showcase_sobel_norm_thresholding()
    # showcase_quadratic_norm_thresholding()
    # showcase_directional_thresholding()
    # showcase_hls_thresholding()
    # showcase_red_channel_thresholding()
    showcase_combined_thresholding()
    print "Works like a charm!!"
