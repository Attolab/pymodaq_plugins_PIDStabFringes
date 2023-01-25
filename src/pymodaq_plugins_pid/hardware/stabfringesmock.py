import numpy as np
# from pymodaq.daq_utils.daq_utils import gauss2D
# from scipy.optimize import curve_fit

'''
Controller for the two mocks.
'''



class StabFringesController:

    axis = ['H', 'V', 'Theta']
    Nactuators = len(axis)
    Nx = 256
    Ny = 256
    offset_x = 0
    offset_y = 128
    coeff = 0.01
    angle = 0
    drift = True

    def __init__(self, positions=None, noise=0.1, amp=10, frq=1/25, drft_spd = 0.01, drft_alea = 0.01, angle=0):
        super().__init__()
        if positions is None:
            self.current_positions = dict(zip(self.axis, [0. for ind in range(self.Nactuators)]))
        else:
            assert isinstance(positions, list)
            assert len(positions) == self.Nactuators
            self.current_positions = positions

        global x000
        x000 = 0
        self.amp = amp
        self.frq = frq
        self.noise = noise
        self.data_mock = None
        self.drft_spd = drft_spd
        self.drft_alea = drft_alea


    def check_position(self, axis):
        return self.current_positions[axis]

    def move_abs(self, position, axis):
        self.current_positions[axis] = position
        global x000
        x000 = position

    def move_rel(self, position, axis):
        self.current_positions[axis] += position
        global x000
        x000 += position

    def get_xaxis(self):
        return np.linspace(0, self.Nx, self.Nx, endpoint=False)

    def get_yaxis(self):
        return np.linspace(0, self.Ny, self.Ny, endpoint=False)

    def set_Mock_data(self):
        """
        """
        global x000
        x_axis = self.get_xaxis()
        y_axis = self.get_yaxis()
        if self.drift:
            self.offset_x += self.drft_spd + self.drft_alea * (0.5-np.random.rand(1))
        drf = (x000 + self.offset_x + self.coeff * self.current_positions['H']) #% (2*np.pi)
        # print(drf)
        self.data_mock = self.sin2D(x_axis, y_axis, drf, self.angle*2*np.pi / 360)
        # print("Camera : ", self.frq, drf % (2*np.pi/self.frq))

        # sm = np.sum(self.data_mock, axis=0)
        # sm -= np.mean(sm)
        # sm /= np.max(np.abs(sm))

        # # popt, pcov = curve_fit(self.sinus, x_axis, sm, p0=np.array([1, 0, 1, 0]), bounds=([0.05, 0, 0.8, -0.5], [10, 6.28, 1.2, 0.5]))
        # popt, pcov = curve_fit(self.sinus, x_axis, sm, p0=np.array([0.05, 0]), method='lm')
        # # popt, pcov = curve_fit(self.sinus, x_axis, sm, p0=np.array([0.05, 0]), bounds=([0.05, 0], [10, 6.28]), method='dogbox')
        # phi = popt[1]
        # self.curr_input = phi

        # # print('input conversion done')
        # print("Fit : ", popt)
        # print("Var : ", np.diag(pcov))
        return self.data_mock

    # def sinus(self, x, frq, phi):
    # # def sinus(self, x, frq, phi, a, b):
    #     # xrot = x*np.cos(theta) + y*np.sin(theta)
    #     # print(a, '\n', frq, '\n', x, '\n', phi, '\n', b)
    #     # print('a = ', type(a))
    #     # print('x = ', type(x))
    #     # print('frq = ', type(frq))
    #     # print('phi = ', type(phi))
    #     # print('b = ', type(b))
    #     return(np.sin(2*np.pi*frq*x+phi))
    #     # return(a * np.sin(frq*x+phi)+b)

    def sin2D(self, x, y, x0, angle_rad):
        Nx = len(x) if hasattr(x, '__len__') else 1
        Ny = len(x) if hasattr(y, '__len__') else 1

        X, Y = np.meshgrid(x, y)


        # dx = self.amp * np.sin((x+x0)*self.frq) 
        # data = np.outer(np.ones(Ny), dx) 
        data = np.sin((np.cos(angle_rad) * (X+x0) + np.sin(angle_rad) * (Y+x0))*self.frq)+ self.noise * np.random.rand(Nx, Ny)
        # data = self.amp * gauss2D(x, x0, self.wh[0], y, y0, self.wh[1], 1, self.current_positions['Theta']) +\
        #        self.noise * np.random.rand(Nx, Ny)

        return data
        # return np.squeeze(data)

    # def gauss2D(self, x, y, x0, y0):
    #     Nx = len(x) if hasattr(x, '__len__') else 1
    #     Ny = len(x) if hasattr(y, '__len__') else 1
    #     data = self.amp * gauss2D(x, x0, self.wh[0], y, y0, self.wh[1], 1, self.current_positions['Theta']) +\
    #            self.noise * np.random.rand(Nx, Ny)

    #     return np.squeeze(data)

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

# def sinus(xe, fr, phi):
#     print(fr, phi)
#     return(np.cos(2*np.pi*fr*xe + phi))
#     # return(a*np.sin(fr*xe+phi)+b)

# def sinusb(xe, phi):
#     print(phi)
#     return(np.sin(2*np.pi*xe + phi))

# import pyqtgraph as pg

# def Axes_TF(t):
#     '''
#     Calcule l'axe dans l'espace correspondant à celui donné. Si on lui donne un temps, il calcule l'axe des fréquences et inversement.
#     '''
# #     dt = t[1]-t[0]
#     n = len(t)
#     tm = t[-1]
#     f = np.linspace(-n/(4*tm), n/(tm*4), n)
#     return(f)

# if __name__ == '__main__':
#     pg.setConfigOption('background', 'w')
#     pg.setConfigOption('foreground', 'w')
#     plotWidget = pg.plot()

#     # s = StabFringesController()
    

#     # s.frq = 0.1
#     # s.phi = 0

#     # im = s.set_Mock_data()
#     # xi = s.get_xaxis()
#     # Nx = len(xi)

#     # sm = np.sum(im, axis=0)
#     # sm -= np.mean(sm)
#     # sm /= np.max(np.abs(sm))



#     # # # popt, pcov = curve_fit(self.sinus, x_axis, sm, p0=np.array([1, 0, 1, 0]), bounds=([0.05, 0, 0.8, -0.5], [10, 6.28, 1.2, 0.5]))
#     # popt, pcov, info, mesg, ier = curve_fit(sinus, xi, sm, p0=np.array([0.05, 0]), full_output=True)
#     # # popt, pcov, info, mesg, ier = curve_fit(sinus, xi, sm, p0=np.array([0.5, 0, 1, 0]), full_output=True, absolute_sigma=True)
#     # # popt, pcov = curve_fit(sinus, xi, sm, p0=np.array([1, 0, 1, 0]), bounds=([0.05, 0, 0, 0], [10, 6.28, 1000, 1000]))
#     # # popt, pcov = curve_fit(sinus, xi, sm, p0=np.array([0.05, 0]), method='lm')
#     # # popt, pcov = curve_fit(sinus, xi, sm, p0=np.array([0.5, 0]), bounds=([0.05, 0], [10, 6.28]), epsfcn=0.1)
#     # # phi = popt[1]
#     # # self.curr_input = phi

#     # # # print('input conversion done')
#     # print("Fit : ", popt)
#     # print("Var : ", np.diag(pcov))

#     # # plotWidget.plot(x, sinus(x, popt[0], popt[1]))
#     # plotWidget.plot(xi, sm, pen='r')
#     # plotWidget.plot(xi, sinus(xi, popt[0], popt[1]), pen='b')
#     # # plotWidget.plot(xi, sinus(xi, popt[0], popt[1], popt[2], popt[3]), pen='b')

#     # # plotWidget.plot(xi, sinus(xi, 0.48, 0.0011), pen='w')
#     # # plotWidget.plot(xi, sinus(xi, 0.1, 0), pen='w')
#     # # plotWidget.plot(xi, sinus(xi, 0.1, 0, 1, 0), pen='w')

#     # # print(info)
#     # print(mesg)
#     # print(ier)

#     xi = np.arange(256)
#     Nx = len(xi)
#     sm = sinus(xi, 0.05, 0)

#     tf = np.fft.rfft(np.fft.fftshift(sm))
#     kx = np.fft.rfftfreq(Nx)

#     # plotWidget.plot(kx, np.abs(tf))
#     fr = kx[np.argmin(-np.abs(tf))]
#     print(fr)
#     # phi, dph, info, mesg, ier = curve_fit(sinusb, fr*xi, sm, p0=1, full_output=True)
#     [frq, phi], dph, info, mesg, ier = curve_fit(sinus, xi, sm, p0=[fr, 0], bounds=([fr*0.9, -np.pi], [fr*1.1, np.pi]), full_output=True, max_nfev=10000)
#     # phi, dph, info, mesg, ier = curve_fit(sinusb, fr*xi, sm, p0=1, bounds=(-np.pi, np.pi), method='dogbox', full_output=True, max_nfev=10000)

#     plotWidget.plot(xi, sm, pen='r')
#     plotWidget.plot(xi, sinus(xi, frq, phi), pen='--b')

#     print(phi, dph)
#     # print(info)
#     print(mesg)
#     print(ier)
