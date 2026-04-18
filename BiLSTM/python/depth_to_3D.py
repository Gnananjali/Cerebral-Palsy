# (c) 2018 Fraunhofer IOSB
# see mini-rgbd-license.txt for licensing information

import numpy as np
import cv2
from os.path import join, exists
from os import makedirs

# camera calibration used for generation of depth
fx = 588.67905803875317
fy = 590.25690113005601
cx = 322.22048191353628
cy = 237.46785983766890

# adjust path if necessary
folder = '../MINI-RGBD_web/01/depth'
filename = 'syn_00000_depth.png'
output_folder = '../MINI-RGBD_web/01/3D'

# load depth image using OpenCV - can be replaced by any other library that loads image to numpy array
depth_im = cv2.imread(join(folder, filename), -1)

# create tuple containing image indices
indices = tuple(np.mgrid[:480,:640].reshape((2,-1)))

pts3D = np.zeros((indices[0].size, 3))
pts3D[:, 2] = depth_im[indices].ravel() / 1000.
pts3D[:, 0] = (np.asarray(indices).T[:, 1] - cx) * pts3D[:, 2] / fx
pts3D[:, 1] = (np.asarray(indices).T[:, 0] - cy) * pts3D[:, 2] / fy

# write to .obj file
if not exists(output_folder):
	makedirs(output_folder)

with open(join(output_folder, 'syn_00000_3D.obj'), 'w') as file:
    for pt in pts3D:
        file.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))

