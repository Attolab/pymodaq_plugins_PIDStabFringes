import numpy as np

'''
Controller for the two mocks.
'''



class StabSpectroController:

    axis = ['Xaxis']
    Nactuators = 1
    Nt = 256
    drift = False
    wl = 8e-7
    omega = 3e8 / wl * (2*np.pi)
    tau = 25

    def __init__(self, positions=None, noise=0.1, drft_spd = 0, drft_alea = 0):
        super().__init__()
        if positions is None:
            self.current_positions = dict(zip(self.axis, [0. for ind in range(self.Nactuators)]))
        else:
            assert isinstance(positions, list)
            assert len(positions) == self.Nactuators
            self.current_positions = positions

        global deltaT
        deltaT = 1e-12


        self.noise = noise
        self.data_mock = None
        self.drft_spd = drft_spd * 1e-15
        self.drft_alea = drft_alea * 1e-15
        self.taxis = np.linspace(-50e-15, 50e-15, self.Nt)
        self.offsett = 0

    def check_position(self, axis):
        return self.current_positions[axis]

    def move_abs(self, position, axis):
        self.current_positions[axis] = position
        global deltaT
        deltaT = position * 1e-7 * self.omega / 3e8

    def move_rel(self, position, axis):
        self.current_positions[axis] += position
        global deltaT
        deltaT += position * 1e-7 / 3e8
        print(deltaT)

    def get_xaxis(self):
        return(np.fft.rfftfreq(self.Nt, self.taxis[1]-self.taxis[0]))
        # return np.linspace(0, self.Nx, self.Nx, endpoint=False)

    def get_yaxis(self):
        return np.linspace(0, self.Ny, self.Ny, endpoint=False)

    def set_Mock_data(self):
        """
        """
        global deltaT
        # print('a')

        if self.drift:
            # print('ab')
            self.offsett += self.drft_spd + self.drft_alea * (0.5-np.random.rand(1))
        # print('b')
        # p1 = self.gaussian(self.taxis, self.omega, self.tau, dt=0)
        # # p1 = self.gaussian(self.taxis, self.omega, self.tau)
        # # print('b1')
        # p2 = self.gaussian(self.taxis, self.omega, self.tau, dt=deltaT+self.offsett)
        # # print('b2')
        # pulses = p1 + p2
        pulses = self.gaussian(self.taxis, self.omega, self.tau, dt=0) + self.gaussian(self.taxis, self.omega, self.tau, dt=deltaT+self.offsett) + self.noise*(np.random.rand(self.Nt)-0.5)
        if 0:
            # print('c')
            self.data_mock = np.abs(pulses)**2
        else:
            # self.data_mock = np.abs(np.fft.fftshift(np.fft.fft(pulses)))**2
            self.data_mock = np.abs(np.fft.rfft(np.abs(np.fft.fftshift(np.fft.fft(pulses)))**2))
        # print('d')
        return self.data_mock

    def gaussian(self, t, omega, tau, dt=0, a=1):
        sig = a*np.exp(1j*omega*(t+dt)) * np.exp(-(t+dt)**2/(tau/2)**2)
        return(sig)


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
            return data
        elif data_dim == '2D':
            return data


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
