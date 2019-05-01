import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint


def z(point, constant, n):
  c = point
  c_numbers = np.array(c)
  for i in range(n):
      c = c**2 + constant
      c_numbers = np.append(c_numbers, c)
  return np.column_stack((c_numbers.real, c_numbers.imag))


def unit_circle():
    radians = np.linspace(-np.pi, np.pi, 1000)
    circle = np.exp(1j * radians)
    return np.column_stack((circle.real, circle.imag))

class MovePoint():

    def __init__(self, ax, start_point, tolerance=1):
        self.ax = ax

        # define complex constant to be added
        self.constant = patches.Circle((0, 0), 0.05, fc='b', alpha=0.5,
            gid='constant')
        self.ax.add_patch(self.constant)
        self.constant.set_picker(tolerance)
        canvas = self.constant.figure.canvas
        canvas.mpl_connect('button_press_event', self.on_press)
        canvas.mpl_connect('button_release_event', self.on_release)
        canvas.mpl_connect('pick_event', self.on_pick)
        canvas.mpl_connect('motion_notify_event', self.on_motion)

        # define point in complex plane
        self.point = patches.Circle((start_point.real, start_point.imag),
            0.05, fc='y', alpha=0.5, gid='point')
        self.ax.add_patch(self.point)
        self.point.set_picker(tolerance)
        canvas = self.point.figure.canvas
        canvas.mpl_connect('button_press_event', self.on_press)
        canvas.mpl_connect('button_release_event', self.on_release)
        canvas.mpl_connect('pick_event', self.on_pick)
        canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.current_object = None
        self.currently_dragging = False
        self.plot_function()

    def on_press(self, event):
        self.currently_dragging = True

    def on_release(self, event):
        self.current_object = None
        self.currently_dragging = False

    def on_pick(self, event):
        self.current_object = event.artist

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_object == None:
            return

        self.current_object.center = event.xdata, event.ydata
        # if self.current_object.get_gid() == 'point':
        self.remove_function_from_plot()
        self.plot_function()

        self.point.figure.canvas.draw()

    def plot_function(self):
        c_real, c_imag = zip(*z(complex(self.point.center[0], self.point.center[1]),
                                complex(self.constant.center[0], self.constant.center[1]),
                                100))
        self.function_plot, = self.ax.plot(c_real, c_imag, 'o', color='r',
            markersize=2)

    def remove_function_from_plot(self):
        try:
            self.function_plot.remove()
        except ValueError:
            pass

def main(start_point, n):
    # set the plot outline, including axes going through the origin
    fig, ax = plt.subplots()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect(1)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    # plot unit circle
    c_real, c_imag = zip(*1*unit_circle())
    ax.plot(c_real, c_imag)

    dr = MovePoint(ax, start_point)

    plt.show()


if __name__ == "__main__":
    # a good example for constant = 0.3 + 0.25j
    start_point = complex(sys.argv[1])
    n = int(sys.argv[2])
    main(start_point, n)
