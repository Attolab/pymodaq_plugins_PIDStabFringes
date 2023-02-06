from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector, main
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import argrelextrema

class PIDModelStabFringes(PIDModelGeneric):
    '''
    PID model to stabilise a fringe pattern.
    '''

    limits = dict(max=dict(state=True, value=180),
                  min=dict(state=True, value=-180),)
    konstants = dict(kp=1.0, ki=0.000, kd=0.0000)

    setpoint_ini = [0]
    setpoints_names = ['Phase']

    actuators_name = ["Move 00"]
    detectors_name = ['Det 00']

    Nsetpoints = 1
    params = [{'title': 'Wavelength (nm)', 'name': 'wavelength', 'type': 'float', 'value': 800},
    {'title': 'Actuator units (m)', 'name': 'unit', 'type': 'float', 'value': 1e-6},
    {'title': 'Show converted', 'name': 'show_converted', 'type': 'bool', 'value': False}]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)
        self.wavelength = self.params[0]['value']
        self.unit = self.params[1]['value']
        # self.show = self.params[2]['value']

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == 'wavelength':
            self.wavelength = param.value()
            print(self.wavelength)
        elif param.name() == 'unit':
            self.unit = param.value()
            print(self.unit)
        # elif param.name() == 'show_converted':
        #     self.show = param.value()
        #     print(self.unit)
        else:
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
        key = list(measurements[self.detectors_name[0]]['data1D'].keys())[0]
        sm =  measurements[self.detectors_name[0]]['data1D'][key]['data']

        tf = np.abs(np.fft.rfft(sm))
        arg = 0
        x_maxs = argrelextrema(tf, np.greater)[0]
        if len(x_maxs) > 0:
            x_max = np.argmin(-tf[x_max])
            if tf[x_max] > tf[0]*0.1:
                arg = x_max
        return InputFromDetector([arg])


    def convert_output(self, outputs, dt, stab=True):
        """
        Convert the calculated phase adjustment into a displacement (in µm).

        Parameters
        ----------
        output: (float) output value from the PID from which the model extract a value of the same units as the actuator

        Returns
        -------
        list: the converted output as a list (if there are a few actuators)

        """
        # print('output converted')
        # print(outputs)
        # if outputs[0] != None:
            # self.curr_output = np.array(outputs) /360 * self.wavelength * 1e-9 /2 / self.unit   #Phase in degree, displacement in µm, wavelength in nm. 
        # else:
        self.curr_output = outputs
        # print(self.curr_output)
        return OutputToActuator(mode='rel', values=self.curr_output)


if __name__ == '__main__':
    main("BeamSteeringMockNoModel.xml")


