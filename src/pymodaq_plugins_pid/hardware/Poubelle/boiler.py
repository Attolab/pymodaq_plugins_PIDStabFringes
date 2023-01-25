from time import perf_counter
from qtpy.QtCore import QObject
from numpy.random import random
import numpy as np
from pymodaq import Q_


class BoilerController(QObject):
    _current_temperature = 20.
    _ambiant_temperature = 19.
    _noise = 0.1

    def __init__(self, ):
        super().__init__()
        self.startTimer(10)
        self._current_power = 0.
        self._ellapsed_time = Q_(0., 's')
        self._tau = Q_(1, 's')

    def timerEvent(self, event):
        dt = perf_counter() - self._ellapsed_time
        self._ellapsed_time += dt

        self._current_temperature += 1 * self._current_power * dt + self._noise * (random() - 0.5)
        # some heat dissipation
        self._current_temperature -= 0.2 * dt
        self._current_temperature = np.clip(self._current_temperature, self.ambiant_temp, None)

    def check_position(self):
        return self._current_power

    def move_abs(self, value):
        self._current_power = value

    @property
    def ambiant_temp(self):
        return self._ambiant_temperature

    @ambiant_temp.setter
    def ambiant_temp(self, temperature):
        self._ambiant_temperature = temperature


    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise

    def move_rel(self, value):
        self._current_power += value

    def grab(self):
        return self._current_temperature
