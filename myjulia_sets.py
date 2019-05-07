import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot ranges
r_min, r_max = -2.0, 2.0
c_min, c_max = -2.0, 2.0
plot_size = 200
max_iterations = 50
cmap='hot'

# Even intervals for points to compute orbits of
r_range = np.arange(r_min, r_max, (r_max - r_min) / plot_size)
c_range = np.arange(c_min, c_max, (c_max - c_min) / plot_size)

constant = -0.624+0.435j
# c = 0.0+0.0j

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


z_func_df = pd.DataFrame(zip(points.real, points.imag, stables, colors),
                         columns=['real', 'imag', 'stable', 'iterations'])

print(len(points), len(stables), len(colors), len(z_func_df))
print(z_func_df.head(5))
print(z_func_df.tail(5))

fig, ax = plt.subplots(figsize=(5, 5))

ax.axis([-2, 2, -2, 2])
ax.set_xlabel('x0')
ax.set_ylabel('c')

ax.scatter(points.real, points.imag, c=colors, cmap=cmap, alpha=1)
ax.set_aspect(1)

plt.show()
