import os
import ra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

folder = 'data/train_img_slices'
filename = '1_0.ra'
#256 x 320
for i in range(2):
    img = ra.read_ra(os.path.join(folder, '1_{}.ra'.format(i)))
    mag = abs(img)

    print(len(mag.shape))

    plt.figure()
    plt.imshow(mag, cmap='gray')
    plt.show()