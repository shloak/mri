import sys 
sys.path.append('..')
import matplotlib.pyplot as plt
from glob import glob
import ra
import numpy as np
import pylab 
import scipy.stats as stats
from utils import *
import tensorflow as tf

def poisson(img_shape, accel, K=30, calib=[0, 0], dtype=np.complex,
            crop_corner=True, return_density=False, seed=0):
    ''' Generate Poisson-disk sampling pattern

    Inputs
    ------
    img_shape - Length-2 array containing x and y dimensions
    a - Target acceleration factor. Greater than 1
    K - maximum number of samples to reject
    calib - calibration size
    '''

    y, x = np.mgrid[:img_shape[-2], :img_shape[-1]]
    x = np.maximum(abs(x - img_shape[-1] / 2) - calib[-1] / 2, 0)
    x /= x.max()
    y = np.maximum(abs(y - img_shape[-2] / 2) - calib[-2] / 2, 0)
    y /= y.max()
    r = np.sqrt(x ** 2 + y ** 2)

    slope_max = 40
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2.0
        R = (1.0 + r * slope)
        mask = _poisson(img_shape[-1], img_shape[-2], K, R, calib, seed)
        if crop_corner:
            mask *= r < 1

        est_accel = img_shape[-1] * img_shape[-2] / np.sum(mask[:])

        if abs(est_accel - accel) < 0.1:
            break
        if est_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    mask = mask.reshape(img_shape).astype(dtype)
    if return_density:
        return mask, r
    else:
        return mask

def _poisson(nx, ny, K, R, calib, seed):
    '''
    img_dims - Length-2 array containing x and y dimensions
    a - Target acceleration factor. Greater than 1
    K - maximum number of samples to reject
    '''
    mask = np.zeros((ny, nx))
    f = ny / nx

    if seed is not None:
        np.random.seed(int(seed))

    pxs = np.empty(nx * ny, np.int32)
    pys = np.empty(nx * ny, np.int32)
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    m = 1
    while (m > 0):

        i = np.random.randint(0, m)
        px = pxs[i]
        py = pys[i]
        rad = R[py, px]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < K:

            # Generate point randomly from R and 2R
            rd = rad * math.sqrt(np.random.random() * 3 + 1)
            t = 2 * math.pi * np.random.random()
            qx = px + rd * math.cos(t)
            qy = py + rd * f * math.sin(t)

            # Reject if outside grid or close to other points
            if qx >= 0 and qx < nx and qy >= 0 and qy < ny:

                startx = max(int(qx - rad), 0)
                endx = min(int(qx + rad + 1), nx)
                starty = max(int(qy - rad * f), 0)
                endy = min(int(qy + rad * f + 1), ny)

                done = True
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[y, x] == 1 and
                            (((qx - x) / R[y, x]) ** 2 +
                             ((qy - y) / (R[y, x] * f)) ** 2 < 1)):
                            done = False
                            break

                    if not done:
                        break

            k += 1

        # Add point if done else remove active
        if done:
            pxs[m] = qx
            pys[m] = qy
            mask[int(qy), int(qx)] = 1
            m += 1
        else:
            pxs[i] = pxs[m - 1]
            pys[i] = pys[m - 1]
            m -= 1

    # Add calibration region
    mask[int(ny / 2 - calib[-2] / 2):int(ny / 2 + calib[-2] / 2),
         int(nx / 2 - calib[-1] / 2):int(nx / 2 + calib[-1] / 2)] = 1

    return mask


rands = np.random.randint(10000, size=20)
subs = [2, 4, 6]
for sub in subs:
    for i in range(20):
        mask = poisson([320, 256], sub, calib=[24, 24], seed=rands[i])
        np.save(("../masks/gen_masks/{}_").format(sub) + str(i), mask)
        print(("../masks/gen_masks/{}_").format(sub) + str(i))