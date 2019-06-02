import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import json
from pprint import pprint

# Plot ranges
r_min, r_max = -1.5, 1.5
c_min, c_max = -1.1, 1.1
dpu = 150  # dots per unit - 50 dots per 1 units means 200 points per 4 units
max_iterations = 50
cmap='nipy_spectral'

# Even intervals for points to compute orbits of
intval = 1 / dpu
r_range = np.arange(r_min, r_max + intval, intval)
c_range = np.arange(c_min, c_max + intval, intval)

# constant = -0.624+0.435j
# constant = 0.0+0.0j
# constant = -0.65 + 0.36j
# constant = -0.75 + 0j # mandelbrot set
# constant = 0 + 1j
# constant = -0.1 + 0.65j
# constant = -0.7 + 0.27015j
constant = -0.76 -0.10j

def juliaset_func(point, constant):
    z = point
    stable = True
    num_iterations = 1
    while stable and num_iterations < max_iterations:
        z = z**2 + constant
        if abs(z) > max(abs(constant), 2):
            stable = False
            return (stable, num_iterations)
        num_iterations += 1

    return (stable, num_iterations)


def find_boundary_juliaset(r_range, c_range, constant, max_iterations):
    stables = np.array([], dtype='bool')
    for imag in c_range:
        for real in r_range:
            stable, _ = juliaset_func(complex(real, imag), constant)
            stables = np.append(stables, stable)

    rows = len(r_range)
    cols = len(c_range)
    start = len(stables)
    stable_field = []
    for _ in range(cols):
        start -= rows
        real_vals = [1 if val == True else 0 for val in stables[start:start+rows]]
        stable_field.append(real_vals)
    stable_field = np.array(stable_field, dtype='int8')

    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='int8')
    stable_boundary = convolve2d(stable_field, kernel, mode='same')

    boundary_points = []
    for col in range(cols):
        for row in range(rows):
            if stable_boundary[col, row] in [1]:
                real_val = r_range[row]
                # invert cols as min imag value is highest col and vice versa
                imag_val = c_range[cols-1 - col]
                boundary_points.append((real_val, imag_val))
            else:
                pass

    return boundary_points


points = np.array([])
colors = np.array([])
stables = np.array([], dtype='bool')
progress = 0
for imag in c_range:
    for real in r_range:
        point = complex(real, imag)
        points = np.append(points, point)
        stable, color = juliaset_func(point, constant)
        stables = np.append(stables, stable)
        colors = np.append(colors, color)
    print(f'{100*progress/len(c_range)/len(r_range):3.2f}% completed\r', end='')
    progress += len(r_range)
print('                             \r', end='')

rows = len(r_range)
start = len(colors)
orig_field = []
stable_field = []
for i_num in range(len(c_range)):
    start -= rows
    real_colors = [color for color in colors[start:start+rows]]
    real_stables = [1 if val == True else 0 for val in stables[start:start+rows]]
    orig_field.append(real_colors)
    stable_field.append(real_stables)
orig_field = np.array(orig_field, dtype='int')
stable_field = np.array(stable_field, dtype='int')

kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
stable_boundary = convolve2d(stable_field, kernel, mode='same')

boundary_points = []
cols, rows = stable_boundary.shape
assert cols == len(c_range), "check c_range and cols"
assert rows == len(r_range), "check r_range and rows"

for col in range(cols):
    for row in range(rows):
        if stable_boundary[col, row] in [1]:
            stable_boundary[col, row] = 1
        else:
            stable_boundary[col, row] = 0

boundary_points =  find_boundary_juliaset(r_range, c_range,
                                          constant, max_iterations)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

ax1.matshow(orig_field, cmap=cmap, interpolation='bilinear')
ax2.matshow(stable_field, cmap=cmap)
ax3.matshow(stable_boundary, cmap=cmap)

x = [point[0] for point in boundary_points]
y = [point[1] for point in boundary_points]
ax4.plot(x, y, 'o', c='r', markersize=0.5)
ax4.set_aspect(1)
# ax2.add_patch(Polygon(boundary_points, closed=True, fill=False, lw=0.5, color='w'))

# boundary_dict = {'boundary': boundary_points}
# with open('myjulia.json', 'wt') as json_file:
#     json.dump(boundary_dict, json_file)

plt.show()
