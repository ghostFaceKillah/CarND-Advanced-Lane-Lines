import numpy as np

Y_METERS_PER_PIXEL = 20. / 720
X_METERS_PER_PIXEL = 3.7 / 700

IMG_W = 1280
IMG_H = 720


def quadratic_fit_to_radius(fit, y):
   return ((1 + (2 * fit[0] * y + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])


def scale_quadratic_fit(fit, y_scale=Y_METERS_PER_PIXEL, x_scale=X_METERS_PER_PIXEL):
    return np.array([fit[0] * x_scale / (y_scale ** 2), 
                     fit[1] * x_scale / y_scale,
                     fit[2] * x_scale])


def calculate_curvature_and_center(left_fit, right_fit):
    left_fit_cr = scale_quadratic_fit(left_fit)
    right_fit_cr = scale_quadratic_fit(right_fit)

    lowest_y = IMG_H - 1
    lowest_y_irl = lowest_y * Y_METERS_PER_PIXEL
    left_curv_rad = quadratic_fit_to_radius(left_fit_cr, lowest_y_irl)
    right_curv_rad = quadratic_fit_to_radius(right_fit_cr, lowest_y_irl)

    y = lowest_y
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    mid_x = (left_x + right_x) / 2.

    distance_from_center_in_px = IMG_W/2. - mid_x
    distance_from_center_in_meters = distance_from_center_in_px * X_METERS_PER_PIXEL

    return left_curv_rad, right_curv_rad, distance_from_center_in_meters
