import sys
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def z(constant, n):
  c = constant
  c_list = np.array(constant)
  for i in range(n):
      c = c**2 + constant
      c_list = np.append(c_list, c)

  return c_list

# TODO implement unit circle
# def unit_circle():

# set the plot outline, including axes going through the origin
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.tick_params(axis='both', which='major', labelsize=6)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

constant = complex(sys.argv[1])
n = int(sys.argv[2])
# constant = 0.3 + 0.25j
c_list = z(constant, n)
c_list = np.column_stack((c_list.real, c_list.imag))
c_real, c_imag = zip(*c_list)

ax.scatter(c_real, c_imag)
ax.plot(c_real, c_imag)
# ax.axis('scaled')
plt.show()
