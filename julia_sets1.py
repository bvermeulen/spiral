"""
This solution is an improved version of an efficient Julia set solver
from:
'Bauckhage C. NumPy/SciPy Recipes for Image Processing:
 Creating Fractal Images. researchgate. net, Feb. 2015.'

from:
https://codereview.stackexchange.com/questions/210271/generating-julia-set
"""
import itertools
from functools import partial
from numbers import Complex
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def douady_hubbard_polynomial(z: Complex,
                              *,
                              c: Complex):
    """
    Monic and centered quadratic complex polynomial
    https://en.wikipedia.org/wiki/Complex_quadratic_polynomial#Map
    """
    return z ** 2 + c


def julia_set(*,
              mapping: Callable[[Complex], Complex],
              min_coordinate: Complex,
              max_coordinate: Complex,
              width: int,
              height: int,
              iterations_count: int = 256,
              threshold: float = 2.) -> np.ndarray:
    """
    As described in https://en.wikipedia.org/wiki/Julia_set
    :param mapping: function defining Julia set
    :param min_coordinate: bottom-left complex plane coordinate
    :param max_coordinate: upper-right complex plane coordinate
    :param height: pixels in vertical axis
    :param width: pixels in horizontal axis
    :param iterations_count: number of iterations
    :param threshold: if the magnitude of z becomes greater
    than the threshold we assume that it will diverge to infinity
    :return: 2D pixels array of intensities
    """
    imaginary_axis, real_axis = np.ogrid[
                        min_coordinate.imag: max_coordinate.imag: height * 1j,
                        min_coordinate.real: max_coordinate.real: width * 1j]
    complex_plane = real_axis + 1j * imaginary_axis

    result = np.ones(complex_plane.shape)

    for _ in itertools.repeat(None, iterations_count):
        mask = np.abs(complex_plane) <= threshold
        if not mask.any():
            break
        complex_plane[mask] = mapping(complex_plane[mask])
        result[~mask] += 1

    return result


if __name__ == '__main__':
    mapping = partial(douady_hubbard_polynomial,
                      c=-0.7 + 0.27015j)  # type: Callable[[Complex], Complex]

    image = julia_set(mapping=mapping,
                      min_coordinate=-1.5 - 1j,
                      max_coordinate=1.5 + 1j,
                      width=800,
                      height=600)
    plt.axis('off')
    plt.imshow(image,
               cmap='nipy_spectral',
               origin='lower')
    plt.show()
