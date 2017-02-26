import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm

from undistort import undistort_factory
from utils import show_transformation_on_test_images
from color_grad import combined_thresholding
from perspective import make_warpers


def find_peaks_alt(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)

    hist_ind = np.argsort(histogram)
    hist_ind = hist_ind[::-1]
    hist_min_dist = 500

    # Search for 2 peaks
    peaks = []
    for ind in hist_ind:
        if len(peaks) == 0:
            peaks.append(ind)
            continue
        if len(peaks) == 1:
            if abs(ind-peaks[0]) > hist_min_dist:
                peaks.append(ind)
                break
    peaks = np.sort(peaks)
    leftx_base, rightx_base = peaks
    return leftx_base, rightx_base


def find_starting_points(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def find_lane(binary_warped, nwindows=9, visualize=False):
    # input image is 'binary warped'
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image

    # Create an output image to draw on and visualize the result
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines

    # leftx_base, rightx_base = find_starting_points(binary_warped)
    leftx_base, rightx_base = find_peaks_alt(binary_warped)

    if visualize:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if visualize:
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2) 
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2) 

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualize 
    # Generate x and y values for plotting
    if visualize:
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit


def find_next_lane(binary_warped, left_fit, right_fit):
    # Skip the sliding windows step once you know where the lines are

    # Now you know where the lines are you have a fit! In the next frame of video you
    # don't need to do a blind search again, but instead you can just search in a
    # margin around the previous line position like this:
     
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 100
    left_lane_inds = (
            (nonzerox > (left_fit[0]*(nonzeroy**2) +
                left_fit[1]*nonzeroy + left_fit[2] - margin)) &
            (nonzerox < (left_fit[0]*(nonzeroy**2) +
                left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = (
            (nonzerox > (right_fit[0]*(nonzeroy**2) +
                right_fit[1]*nonzeroy + right_fit[2] - margin)) &
            (nonzerox < (right_fit[0]*(nonzeroy**2) +
                right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit



def illustrate_next_lane_pixel_fit(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def draw_fit_on_original(img, left_fit, right_fit):
    _, unwarp = make_warpers()

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result


def show_line_pixel_finding_one():
    undst = undistort_factory()
    warp, unwarp = make_warpers()
    fname = 'img/test1.jpg'

    img = cv2.imread(fname)
    binary_warped = warp(combined_thresholding(undst(img)))

    left_fit, right_fit = find_lane(binary_warped)

    for img_fname in tqdm.tqdm(glob.glob('img/*')):
        img = cv2.imread(img_fname)
        binary_warped = warp(combined_thresholding(undst(img)))
        find_next_lane(binary_warped, left_fit, right_fit)


def show_line_finding():
    undst = undistort_factory()
    warp, unwarp = make_warpers()

    def process(img):
        binary_warped = warp(combined_thresholding(undst(img)))
        left_fit, right_fit = find_lane(binary_warped)
        return draw_fit_on_original(img, left_fit, right_fit)

    show_transformation_on_test_images(process, "lane finding")


if __name__  == '__main__':
    show_line_finding()
