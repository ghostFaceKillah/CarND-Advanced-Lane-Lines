#!/usr/bin/python
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import ipdb
from moviepy.editor import VideoFileClip

from color_grad import combined_thresholding
from curvature import calculate_curvature_and_center
from finding_lines import draw_fit_on_original, find_lane, find_next_lane
from perspective import make_warpers
from undistort import undistort_factory


class Processor(object):
    def __init__(self, fit_time_decay=0.85):
        self.undistort = undistort_factory()
        self.thresholding = combined_thresholding
        self.warp, self.unwarp = make_warpers()
        self.l_fit = None
        self.r_fit = None
        self.alpha = fit_time_decay
        self.l_curv_hist = []
        self.r_curv_hist = []
        self.center_dist_hist = []
        self.l_curv = 0.
        self.r_curv = 0.
        self.center_dist = None

    def update_fit(self, new_left_fit, new_right_fit):
        a = self.alpha
        self.l_fit = a * self.l_fit + (1 - a) * new_left_fit
        self.r_fit = a * self.r_fit + (1 - a) * new_right_fit

    def update_curvature_stats(self):
        # Update the curvature and center-of-line stats
        stats = calculate_curvature_and_center(self.l_fit, self.r_fit)
        l_curv, r_curv, center_dist = stats

        a = 0.9
        self.l_curv = a * self.l_curv + (1 - a) * l_curv
        self.r_curv = a * self.r_curv + (1 - a) * r_curv
        self.center_dist = center_dist

        self.l_curv_hist.append(stats[0])
        self.r_curv_hist.append(stats[1])
        self.center_dist_hist.append(stats[2])

    def dump_curvature_stats(self):
        df = pd.DataFrame({
            'left': self.l_curv_hist,
            'right': self.r_curv_hist,
            'distance from center': self.center_dist_hist
        })
        df.to_csv('stats.csv')
        np.clip(df[['left', 'right']], 0, 2000).plot()
        plt.title('Lane line implied radius of lane curvature')
        plt.savefig('out/curvatures.jpg')
        plt.close()

        df[['distance from center']].plot()
        plt.title('Implied distance of the car from center')
        plt.savefig('out/dist_from_center.jpg')
        plt.close()

    def draw_curvature_center_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = "Left rad. of curv = {:.2f} m, right rad. of curv = {:.2f} m, "\
              "distance from center = {:.2f} m"
        f = lambda x: np.round(x / 10.) * 10.
        txt = txt.format(f(self.l_curv), f(self.r_curv), self.center_dist)
        out = cv2.putText(img, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (0,0,0), 1, cv2.LINE_AA)
        return out

    def process_image(self, img):
        undistorted = self.undistort(img)
        binary = self.thresholding(undistorted)
        binary_warped = self.warp(binary)

        if self.l_fit is None:
            left_fit, right_fit = find_lane(binary_warped)
            self.l_fit = left_fit
            self.r_fit = right_fit
        else:
            l_fit, r_fit = find_next_lane(binary_warped, self.l_fit, self.r_fit)
            self.update_fit(l_fit, r_fit)

        self.update_curvature_stats()

        out = draw_fit_on_original(undistorted, self.l_fit, self.r_fit)
        out = self.draw_curvature_center_info(out)
        return out


def main():
    parser = argparse.ArgumentParser(description="Find Lane Lines on a video")
    parser.add_argument('--in_video', type=str, help='input video', required=True)
    parser.add_argument('--out_video', type=str, help='output video', required=True)

    args = parser.parse_args()

    video_file = args.in_video
    output_video_file = args.out_video

    clip = VideoFileClip(video_file)

    processor = Processor()

    clip = clip.fl_image(processor.process_image)
    clip.write_videofile(output_video_file, audio=False)

    processor.dump_curvature_stats()


if __name__ == '__main__':
    main()
