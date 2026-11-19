# Useful references

## Classic feature matching (OpenCV)

- Feature matching + homography to find objects: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
- Feature matching (BFMatcher, FLANN, ratio test): https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
- SIFT introduction: https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
- ORB: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
- `cv2.findHomography` / `cv2.perspectiveTransform`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- CLAHE (contrast equalization): https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
- Image pyramids / resizing: https://docs.opencv.org/4.x/dc/dff/tutorial_py_pyramids.html

## Deep learning matchers (bonus)

- LightGlue (SuperPoint / DISK / ALIKED + LightGlue): https://github.com/cvg/LightGlue
- LoFTR in Kornia: https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LoFTR
- Kornia feature matching tutorials: https://kornia.github.io/tutorials/

## Trajectory and visualization

- SciPy Savitzky-Golay filter (smoothing): https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
- SciPy spline interpolation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html
- OpenCV drawing functions: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
- OpenCV VideoWriter: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html

## Tips

- Map is large: downscale it (0.25x to 0.5x) and frames to a similar ground resolution before matching.
- Use Lowe's ratio test (about 0.7 to 0.8) and RANSAC (`cv2.RANSAC`, threshold about 5 px).
- Consecutive frames are close to each other: once frame N is found, search frame N+1 in a window around it first.
- Reject a frame if inliers are few (for example under 15) and fill its position by interpolation.
