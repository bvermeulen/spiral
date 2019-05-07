import numpy as np
from scipy.signal import convolve2d
import pandas as pd
import matplotlib.pyplot as plt
from  shapely.geometry import Polygon, Point
from matplotlib.collections import PatchCollection
from pprint import pprint

# Plot ranges
r_min, r_max = -2.0, 2.0
c_min, c_max = -2.0, 2.0
plot_size = 20
max_iterations = 50
cmap='coolwarm'

# Even intervals for points to compute orbits of
r_range = np.arange(r_min, r_max, (r_max - r_min) / plot_size)
c_range = np.arange(c_min, c_max, (c_max - c_min) / plot_size)

# constant = -0.624+0.435j
constant = 0.0+0.0j

def z_func(point, constant):
    z = point
    stable = True
    num_iterations = 1
    while stable and num_iterations < max_iterations:
        z = z**2 + constant
        if abs(z) > max(abs(constant), 2):
            stable = False
            return (stable, num_iterations)
        num_iterations += 1

    return (stable, 0)

points = np.array([])
colors = np.array([])
stables = np.array([], dtype='bool')
progress = 0
for imag in c_range:
    for real in r_range:
        point = complex(real, imag)
        points = np.append(points, point)
        stable, color = z_func(point, constant)
        stables = np.append(stables, stable)
        colors = np.append(colors, color)
    print(f'{100*progress/len(c_range)/len(r_range):3.2f}% completed\r', end='')
    progress += len(r_range)

# z_func_df = pd.DataFrame(zip(points.real, points.imag, stables, colors),
#                          columns=['real', 'imag', 'stable', 'iterations'])

print('                             \r', end='')

rows = len(r_range)
start = len(colors)
orig_field = []
for i_num in range(len(c_range)):
    start -= rows
    real_vals = [color for color in colors[start:start+rows]]
    orig_field.append(real_vals)
orig_field = np.array(orig_field, dtype='int')

pprint(orig_field)

rows = len(r_range)
start = len(stables)
stable_field = []
for i_num in range(len(c_range)):
    start -= rows
    real_vals = [1 if val == True else 0 for val in stables[start:start+rows]]
    stable_field.append(real_vals)
stable_field = np.array(stable_field, dtype='int')

kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
stable_boundary = convolve2d(stable_field, kernel, mode='same')

boundary_points = []
rows, cols = stable_boundary.shape
zero_field = np.zeros((cols, rows))
for col in range(cols):
    for row in range(rows):
        if stable_boundary[col, row] in range(1,2):
            real_val = -r_range[row]
            imag_val = c_range[col]
            stable_boundary[col, row] = 1
            boundary_points.append((real_val, imag_val))
        else:
            stable_boundary[col, row] = 0

# # sort the bounadary polygon
# n_bp = len(boundary_points)
# centre=(sum([p[0] for p in boundary_points])/n_bp,
#         sum([p[1] for p in boundary_points])/n_bp)
# print(f'centre : {centre}')
# # sort by polar angle
# boundary_points.sort(key=lambda p: np.arctan2(p[0]-centre[0],p[1]-centre[1]))
# boundary_points = np.array(boundary_points)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

ax1.matshow(orig_field, cmap=cmap)
ax2.matshow(stable_field, cmap=cmap)
ax3.matshow(stable_boundary, cmap=cmap)

x = [point[0] for point in boundary_points]
y = [point[1] for point in boundary_points]
ax4.plot(x, y, 'o', c='r', markersize=0.5)
ax4.set_aspect(1)
# ax2.add_patch(Polygon(boundary_points, closed=True, fill=False, lw=0.5, color='w'))


plt.show()
