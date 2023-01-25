from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector, main
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
import numpy as np

class PIDModelStabFringes(PIDModelGeneric):
    '''
    PID model to stabilise a fringe pattern.
    '''

    limits = dict(max=dict(state=True, value=10),
                  min=dict(state=True, value=-10),)
    konstants = dict(kp=10, ki=0.000, kd=0.1000)

    setpoint_ini = [0]
    setpoints_names = ['Xaxis']

    actuators_name = ["Move"]
    detectors_name = ['Det']

    Nsetpoints = 1
    # params = [{'title': 'Threshold', 'name': 'threshold', 'type': 'float', 'value': 10.}]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == '':
            pass

    def ini_model(self):
        super().ini_model()

    def convert_input(self, measurements):
        """
        Retrieves a phase from a sinus on an image to send it into a PID.
        Parameters
        ----------
        measurements: (Ordereddict) Ordereded dict of data from a camera
        If there are ROIs, it will take the first array in the DATA1D (corresponding to the vertical sum) 

        Returns
        -------
        float: the phase retrieved (in degrees)

        """

        if len(measurements['Det']['data1D']) > 1:
            key = list(measurements['Det']['data1D'].keys())[0]
            # print(key)
            sm =  measurements['Det']['data1D'][key]['data']
        else:
            # print('No ROI defined, we will sum vertically.')
            key = list(measurements['Det']['data2D'].keys())[0]  # so it can also be used from another plugin having another key
            # print(key)
            image = measurements['Det']['data2D'][key]['data']
            sm = np.sum(image, axis=0)

        sm = sm - np.mean(sm)
        sm = sm / np.max(np.abs(sm))

        tf = np.fft.rfft(np.fft.fftshift(sm))
        phi = np.angle(tf[np.argmin(-np.abs(tf))])

        # self.curr_input = phi     #Je ne sais pas à quoi ça sert de l'avoir en self.

        # xi = np.arange(len(sm))   #Tentative de fit, mais en fait c'est pas nécessaire.
        # Nx = len(xi)
        # kx = np.fft.rfftfreq(Nx)
        # fr = kx[np.argmin(-np.abs(tf))]
        # [frq, phi], _ = curve_fit(self.sinus, xi, sm, p0=[fr, 0], bounds=([fr*0.9, -np.pi], [fr*1.1, np.pi]), max_nfev=10000)

        phi *= 180 / np.pi
        # print('input conversion done')
        # print("Fit : ", fr, phi)
        return InputFromDetector([phi])

    # def sinus(self, x, frq, phi): #servait au fit
    #     return(np.sin(frq*x+phi))

    def convert_output(self, outputs, dt, stab=True):       #Je ne sais pas si je suis censé lui donner quelque chose à faire...
        """
        Convert the output of the PID in units to be fed into the actuator
        Parameters
        ----------
        output: (float) output value from the PID from which the model extract a value of the same units as the actuator

        Returns
        -------
        list: the converted output as a list (if there are a few actuators)

        """
        # print('output converted')
        # print(outputs)
        self.curr_output = outputs
        return OutputToActuator(mode='rel', values=outputs)


if __name__ == '__main__':
    main("BeamSteeringMockNoModel.xml")


