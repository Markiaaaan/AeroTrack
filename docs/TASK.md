# Task: UAV route reconstruction (It-Jims CV test task 2026)

Full original description: [It-Jims_CV_test_task_2026.pdf](It-Jims_CV_test_task_2026.pdf)

## Goal

We get a panoramic satellite map of an area and 50 consecutive "photos" taken by a UAV
flying over it (imitated with satellite imagery at a different angle and scale).
We must find where each photo was taken on the map and reconstruct and visualise the
drone's route. Classic (SIFT/ORB) or deep learning (SuperPoint, LightGlue, LoFTR)
feature extraction and matching are both allowed.

## Input data

- Download: https://drive.google.com/file/d/1h6xfYQfokDo5AzSv1Q-x66HluDpWSh87/view?usp=sharing
- Contents: one panoramic satellite map + 50 consecutive UAV frames.
- Put it under `data/` locally. `data/` is git-ignored, do not commit the images.

## Levels

**Standard level (main target)**
Match the 50 frames to the map and reconstruct and visualise the route.

**Advanced level (bonus)**
Open the map full screen, film the screen with a phone or webcam to imitate a flyover,
extract frames from that video and run them through the same pipeline.

## Requirements

- The code reads the region map and the series of frames from paths given as arguments.
- Output is either a single image with the trajectory, or a video with the trajectory
  and the UAV's position moving in real time.

## Deliverables

- [ ] Well organized and commented code
- [ ] `requirements.txt` or `environment.yml`
- [ ] `README.md` with a basic description of the solution
- [ ] An image of the map with the visualised route, showing start, finish and the order of route points

## Notes from the organizers

- Experiment with image downscaling for better feature matching.
- Use trajectory interpolation and smoothing if needed.
- Optional: a video with a zoomed-in map region and per-frame trajectory.
- Optional: draw the bounding boxes of the drone photos on the map, keeping their size and rotation.

## Team plan

The work is split into 4 roles, tracked as GitHub issues:

1. Data and preprocessing
2. Feature matching (core)
3. Trajectory and visualization
4. Integration, docs and advanced level

See [REFERENCES.md](REFERENCES.md) for useful tutorials and libraries.
