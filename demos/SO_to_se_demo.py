import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import li_utils
import draw_utils


fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

ax.set_xlim([-1,1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax_w1 = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_w2 = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_w3 = plt.axes([0.25, 0.3, 0.65, 0.03])

w1_slider = Slider(
    ax=ax_w1,
    label='w_1',
    valmin=-2*np.pi,
    valmax=2*np.pi,
    valinit=0.0,
)

w2_slider = Slider(
    ax=ax_w2,
    label='w_2',
    valmin=-2*np.pi,
    valmax=2*np.pi,
    valinit=0.0,
)

w3_slider = Slider(
    ax=ax_w3,
    label='w_3',
    valmin=-2*np.pi,
    valmax=2*np.pi,
    valinit=0.0,
)

w = np.array([0.0, 0.0, 0.00001])
R = li_utils.rotation_angles_to_matrix(w)
draw_utils.draw_rotation_matrix(ax, R)

def update(val):
    ax.cla()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    w = np.array([w1_slider.val, w2_slider.val, w3_slider.val])
    w1_slider.val = 3.14
    R = li_utils.rotation_angles_to_matrix(w)
    draw_utils.draw_rotation_matrix(ax, R)


w1_slider.on_changed(update)
w2_slider.on_changed(update)
w3_slider.on_changed(update)

plt.show()