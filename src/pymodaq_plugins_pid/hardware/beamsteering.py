import numpy as np
from pymodaq.daq_utils.daq_utils import gauss2D

class BeamSteeringController:

    axis = ['H', 'V', 'Theta']
    Nactuators = len(axis)
    Nx = 256
    Ny = 256
    offset_x = 128
    offset_y = 128
    coeff = 0.01
    drift = False

    def __init__(self, positions=None, wh=(10, 50), noise=0.1, amp=10):
        super().__init__()
        if positions is None:
            self.current_positions = dict(zip(self.axis, [0. for ind in range(self.Nactuators)]))
        else:
            assert isinstance(positions, list)
            assert len(positions) == self.Nactuators
            self.current_positions = positions

        self.amp = amp
        self.noise = noise
        self.wh = wh
        self.data_mock = None

    def check_position(self, axis):
        return self.current_positions[axis]

    def move_abs(self, position, axis):
        self.current_positions[axis] = position

    def move_rel(self, position, axis):
        self.current_positions[axis] += position

    def get_xaxis(self):
        return np.linspace(0, self.Nx, self.Nx, endpoint=False)

    def get_yaxis(self):
        return np.linspace(0, self.Ny, self.Ny, endpoint=False)

    def set_Mock_data(self):
        """
        """
        x_axis = self.get_xaxis()
        y_axis = self.get_yaxis()
        if self.drift:
            self.offset_x += 0.1
            self.offset_y += 0.05
        self.data_mock = self.gauss2D(x_axis, y_axis,
                                 self.offset_x + self.coeff * self.current_positions['H'],
                                 self.offset_y + self.coeff * self.current_positions['V'])
        return self.data_mock

    def gauss2D(self, x, y, x0, y0):
        Nx = len(x) if hasattr(x, '__len__') else 1
        Ny = len(x) if hasattr(y, '__len__') else 1
        data = self.amp * gauss2D(x, x0, self.wh[0], y, y0, self.wh[1], 1, self.current_positions['Theta']) +\
               self.noise * np.random.rand(Nx, Ny)

        return np.squeeze(data)

    def get_data_output(self, data=None, data_dim='0D', x0=128, y0=128, integ='vert'):
        """
        Return generated data (2D gaussian) transformed depending on the parameters
        Parameters
        ----------
        data: (ndarray) data as outputed by set_Mock_data
        data_dim: (str) either '0D', '1D' or '2D'
        x0: (int) if type is '0D" then get value of computed data at this position
        y0: (int) if type is '0D" then get value of computed data at this position
        integ: (str) either 'vert' or 'hor'. Valid if data_dim is '1D" then get value of computed data integrated either
            vertically or horizontally

        Returns
        -------
        numpy nd-array
        """
        if data is None:
            data = self.set_Mock_data()
        if data_dim == '0D':
            return np.array([data[x0, y0]])
        elif data_dim == '1D':
            return np.mean(data, 0 if integ == 'vert' else 1)
        elif data_dim == '2D':
            return data
