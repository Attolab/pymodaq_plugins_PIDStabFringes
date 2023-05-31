from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector, main
from pyqtgraph.dockarea import Dock
from pyqtgraph import InfiniteLine
from pymodaq.daq_utils.plotting.viewer0D.viewer0D_main import Viewer0D
from pymodaq.daq_utils.plotting.viewer2D.viewer_2D_main import Viewer2D
from PyQt5 import QtWidgets
from scipy.ndimage import center_of_mass
import logging
from pymodaq.daq_utils.daq_utils import Axis, set_logger, get_module_name, DataFromPlugins, ThreadCommand
import numpy as np

# Small modifications of BeamSteering model to stabilize a beam. Notably, sets a threshold of minimum intensity and turns off
# PID when it gets too low.
logger = set_logger(get_module_name(__file__))
pid_logger = logging.getLogger('pymodaq.pid_controller')

class PIDModelBeamSteering(PIDModelGeneric):
    limits = dict(max=dict(state=True, value=100),
                  min=dict(state=True, value=-100),)
    konstants = dict(kp=10, ki=0.000, kd=0.1000)

    setpoint_ini = [128, 128]
    setpoints_names = ['Xaxis', 'Yaxis']

    actuators_name = ["Xpiezo", "Ypiezo"]
    detectors_name = ['Camera']

    Nsetpoints = 2
    params = [{'title': 'Noise Threshold', 'name': 'threshold', 'type': 'float', 'value': 10.},
              {'title':'Minimum intensity for correction', 'name':'min_int', 'type': 'float', 'value': 10.}]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)
        self.setupUI()

    def setupUI(self):
        self.dock_camera = Dock("Camera data")
        widget_camera = QtWidgets.QWidget()
        logger.info('Init Widget: Dock Camera')
        self.camera_viewer = Viewer2D(widget_camera)
        self.dock_camera.addWidget(widget_camera)
        self.pid_controller.dock_area.addDock(self.dock_camera)
        self.camera_viewer.show_data(DataFromPlugins(data=[np.random.normal(size=(100, 100))], labels='Camera Image'))

        self.dock_signal = Dock('Integrated Signal')
        widget_signal = QtWidgets.QWidget()
        self.signal_viewer = Viewer0D(widget_signal)
        self.dock_signal.addWidget(widget_signal)
        self.pid_controller.dock_area.addDock(self.dock_signal, 'right', self.dock_camera)
        # self.dock_signal.show_data([np.random.normal(size=100)])

        # Line for threshold
        self.signal_threshold = InfiniteLine(self.settings.child('min_int').value(), angle=0, movable=True)
        self.signal_viewer.ui.Graph1D.plotItem.addItem(self.signal_threshold)
        self.signal_threshold.sigPositionChangeFinished.connect(self.update_signal_threshold)

    def update_signal_threshold(self, line):
        pos = line.value()
        self.settings.child('min_int').setValue(
            pos
        )


    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == 'min_int':
            self.signal_threshold.sigPositionChangeFinished.disconnect(self.update_signal_threshold)
            self.signal_threshold.setPos(param.value())
            self.signal_threshold.sigPositionChangeFinished.connect(self.update_signal_threshold)

    def ini_model(self):
        super().ini_model()

    def convert_input(self, measurements):
        """
        Convert the measurements in the units to be fed to the PID (same dimensionality as the setpoint)
        Parameters
        ----------
        measurements: (Ordereddict) Ordereded dict of object from which the model extract a value of the same units as the setpoint

        Returns
        -------
        float: the converted input

        """
        #print('input conversion done')
        key = list(measurements['Camera']['data2D'].keys())[0]  # so it can also be used from another plugin having another key
        image = measurements['Camera']['data2D'][key]['data']
        image = image - self.settings.child('threshold').value()
        image[image < 0] = 0

        self.camera_viewer.show_data(DataFromPlugins(data=[image]))

        signal = image.sum()
        self.signal_viewer.show_data([[signal]])

        if signal > self.settings.child('min_int').value():
            # Restart PID in case it was stopped by fft peak too low
            if not self.pid_controller.pause_action.isChecked():
                pid_logger.disabled = True  # avoid having hundreds of messages in log
                self.pid_controller.command_pid.emit(ThreadCommand('pause_PID', [False]))
                pid_logger.disabled = False

            x, y = center_of_mass(image)
            self.curr_input = [y, x]
            return InputFromDetector([y, x])

        else:
            # Force the PID to do nothing
            pid_logger.disabled = True  # avoid having hundreds of messages in log
            self.pid_controller.command_pid.emit(ThreadCommand('pause_PID', [True]))
            pid_logger.disabled = False

            return InputFromDetector(self.pid_controller.setpoints)

    def convert_output(self, outputs, dt, stab=True):
        """
        Convert the output of the PID in units to be fed into the actuator
        Parameters
        ----------
        output: (float) output value from the PID from which the model extract a value of the same units as the actuator

        Returns
        -------
        list: the converted output as a list (if there are a few actuators)

        """
        #print('output converted')
        
        self.curr_output = outputs
        return OutputToActuator(mode='rel', values=outputs)


if __name__ == '__main__':
    main("preset_default.xml")


