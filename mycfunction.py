import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pprint import pprint


PLOT_LIMIT = 3
NUM_ITERATIONS = 100
FIG_SIZE = (10, 10)

def z(point, constant):
  c = point
  c_numbers = np.array(c)
  for _ in range(NUM_ITERATIONS):
      c = c**2 + constant
      c_numbers = np.append(c_numbers, c)
  return c_numbers


def unit_circle():
    radians = np.linspace(-np.pi, np.pi, 1000)
    circle = np.exp(1j * radians)
    return np.column_stack((circle.real, circle.imag))

class PlotComplexFunction():

    def __init__(self, start_point, tolerance=1):
        self.fig, self.ax = self.setup_map()
        self.current_object = None
        self.currently_dragging = False
        self.plot_types = ['-o', 'o']
        self.plot_type = 0
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # define a complex constant to be used in complex function
        # this point is blue and can be moved interactively. Initial value is
        # the origin
        self.constant = patches.Circle((0, 0), 0.05, fc='b', alpha=1,
            gid='constant')
        self.ax.add_patch(self.constant)
        self.constant.set_picker(tolerance)
        cv_constant = self.constant.figure.canvas
        cv_constant.mpl_connect('button_press_event', self.on_press)
        cv_constant.mpl_connect('button_release_event', self.on_release)
        cv_constant.mpl_connect('pick_event', self.on_pick)
        cv_constant.mpl_connect('motion_notify_event', self.on_motion)

        # define a starting point in complex plane
        # this point is yellow and can be move interactively
        self.point = patches.Circle((start_point.real, start_point.imag),
            0.05, fc='yellow', alpha=1, gid='point')
        self.ax.add_patch(self.point)
        self.point.set_picker(tolerance)
        cv_point = self.point.figure.canvas
        cv_point.mpl_connect('button_press_event', self.on_press)
        cv_point.mpl_connect('button_release_event', self.on_release)
        cv_point.mpl_connect('pick_event', self.on_pick)
        cv_point.mpl_connect('motion_notify_event', self.on_motion)

        self.plot_function()

    def setup_map(self):
        # set the plot outline, including axes going through the origin
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.set_xlim(-PLOT_LIMIT, PLOT_LIMIT)
        ax.set_ylim(-PLOT_LIMIT, PLOT_LIMIT)
        ax.set_aspect(1)
        ax.set_xticks(np.arange(-PLOT_LIMIT, PLOT_LIMIT+0.1, step=0.2))
        ax.set_yticks(np.arange(-PLOT_LIMIT, PLOT_LIMIT+0.1, step=0.2))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

        # plot unit circle
        c_real, c_imag = zip(*1*unit_circle())
        ax.plot(c_real, c_imag)

        return fig, ax

    def get_ax(self):
        return self.ax

    # TODO this function can probably be removed, to be checked
    def on_press(self, event):
        pass

    def on_release(self, event):
        self.current_object = None
        self.currently_dragging = False

    def on_pick(self, event):
        self.currently_dragging = True
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
        print(f'constant: {self.constant.center[0]:.2f}, {self.constant.center[1]:.2f}\r', end='')
        c_numbers = z(complex(self.point.center[0], self.point.center[1]),
                      complex(self.constant.center[0], self.constant.center[1]))
        self.function_plot, = self.ax.plot(c_numbers.real, c_numbers.imag,
            self.plot_types[self.plot_type], color='r', lw=0.3, markersize=2)

    def remove_function_from_plot(self):
        try:
            self.function_plot.remove()
        except ValueError:
            pass

    def on_key(self, event):
        # with 'space' toggle between just points or points connected with
        # lines
        if event.key == ' ':
            self.plot_type = (self.plot_type + 1) % 2
            self.remove_function_from_plot()
            self.plot_function()
            self.point.figure.canvas.draw()

def main(start_point):
    _ = PlotComplexFunction(start_point)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # a good example for constant = 0.3 + 0.25j
    try:
        start_point = complex(sys.argv[1])
    except IndexError:
        start_point= 0.3 + 0.25j

    main(start_point)
