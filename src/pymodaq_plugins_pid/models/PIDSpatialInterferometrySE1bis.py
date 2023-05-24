from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from pyqtgraph.dockarea import Dock
import scipy.signal.windows as windows
import scipy.fft as fft
from collections import deque
from datetime import datetime
import os, time

from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from scipy.interpolate import interp1d
from pymodaq.daq_utils.daq_utils import linspace_step
from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector
from pymodaq.daq_utils.plotting.viewer0D.viewer0D_main import Viewer0D
from pymodaq.daq_utils.plotting.viewer1D.viewer1D_main import Viewer1D
from pymodaq.daq_utils.plotting.viewer2D.viewer_2D_main import Viewer2D
from pymodaq.daq_utils.daq_utils import Axis, set_logger, get_module_name, DataFromPlugins
from pymodaq.daq_utils.math_utils import ft, ift
from pymodaq.daq_utils.h5modules import H5Saver

from pymodaq.daq_scan import DAQ_Scan

logger = set_logger(get_module_name(__file__))


class PIDModelSpetralInterferometrySE1bis(PIDModelGeneric):
    limits = dict(max=dict(state=False, value=1),
                  min=dict(state=False, value=-1), )
    konstants = dict(kp=0.3, ki=0.0, kd=0.0)

    Nsetpoints = 1
    setpoint_ini = [0]
    setpoints_names = ['Stabilized Delay']

    actuators_name = ['Pump Delay']
    detectors_name = ['Camera PID']

    params = [
        {'title': 'Wavelength (nm)', 'name': 'wavelength', 'type': 'float', 'value': 790},
        {'title': 'Actuator unit (m)', 'name': 'unit', 'type': 'float', 'value': -1e-6},
        {'title': 'Units', 'name': 'unitsGroup', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'Convert to femto', 'name': 'convertFemto', 'type': 'bool', 'value': True},
             {'title': 'Actuator unit (m)', 'name': 'actUnits', 'type': 'float', 'value': -1e-6},
             {'title': 'Set delay to zero', 'name': 'setDelayToZero', 'type': 'bool_push', 'value': False}
         ]},
        {'title': 'Stats', 'name': 'statsGroup', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'RMS', 'name': 'RMS', 'type': 'float', 'value': 0.0, 'readonly': True},
             {'title': 'Record delays', 'name': 'record', 'type': 'bool_push', 'value': False},
             {'title': 'Stop recording', 'name': 'recordStop', 'type': 'bool_push', 'value': False, 'visible': False},
             {'title': 'Number of delays', 'name': 'N_record', 'type': 'int', 'value': 1e3},
         ]},

        {'title': 'Spectrum ROI', 'name': 'spectrum', 'type': 'group', 'expanded': False, 'visible': True,
         'children': [
             {'title': 'Omega min', 'name': 'omega_min', 'type': 'float', 'value': 0.0},
             {'title': 'Omega max', 'name': 'omega_max', 'type': 'float', 'value': 1.0}]},
        {'title': 'Inverse Fourier', 'name': 'inverse_fourier', 'type': 'group', 'expanded': False, 'visible': True,
         'children': [
             {'title': 'N sampling (power of 2)', 'name': 'N_samp_power', 'type': 'float', 'value': 13},
             {'title': 'Centering', 'name': 'centering', 'type': 'bool', 'value': True},
             {'title': 'ROI', 'name': 'ifft_roi', 'type': 'group', 'expanded': True, 'visible': True,
              'children': [
                  {'title': 't min', 'name': 't_min', 'type': 'float', 'value': 0.0},
                  {'title': 't max', 'name': 't_max', 'type': 'float', 'value': 1.0}]}]},
        {'title': 'Show plots', 'name': 'show_plots', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'Show camera data', 'name': 'show_camera', 'type': 'bool', 'value': True},
             {'title': 'Show ROI', 'name': 'show_roi', 'type': 'bool', 'value': True},
             {'title': 'Show FFT', 'name': 'show_fft', 'type': 'bool', 'value': True},
             {'title': 'Show phase', 'name': 'show_phase', 'type': 'bool', 'value': True}]}

    ]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)

        self.wavelength = self.settings.child('wavelength').value()
        self.recording = False
        self.h5saver = H5Saver()
        self.phase_arrays = None
        self.data_channels_initialized = False
        self.setupUI()

    def setupUI(self):
        self.dock_camera = Dock("Camera data")
        widget_camera = QtWidgets.QWidget()
        logger.info('Init Widget: Dock Camera')
        self.camera_viewer = Viewer2D(widget_camera)
        self.dock_camera.addWidget(widget_camera)
        self.pid_controller.dock_area.addDock(self.dock_camera)
        self.camera_viewer.show_data(DataFromPlugins(data=[np.random.normal(size=(100, 100))], labels='Camera Image'))
        self.camera_viewer.view.get_action('ROIselect').trigger()
        self.camera_viewer.view.ROIselect.setPen(pen=(5, 30))
        self.camera_viewer.view.ROIselect.setSize([100, 100])

        # Plot des franges
        self.dock_fringes = Dock('Fringes')
        widget_fringes = QtWidgets.QWidget()
        self.fringe_viewer = Viewer1D(widget_fringes)
        self.dock_fringes.addWidget(widget_fringes)
        self.pid_controller.dock_area.addDock(self.dock_fringes, 'right', self.dock_camera)
        self.fringe_viewer.show_data([np.random.normal(size=100)], labels=['Fringe lineout'])
        logger.info('Init Widget: Dock ROI')

        # Plot de la TF des franges
        self.dock_tf = Dock('Fourier transform')
        widget_tf = QtWidgets.QWidget()
        self.tf_viewer = Viewer1D(widget_tf)
        self.dock_tf.addWidget(widget_tf)
        self.pid_controller.dock_area.addDock(self.dock_tf, 'bottom', self.dock_fringes)
        self.tf_viewer.show_data([np.random.normal(size=100)], labels=['Fourier Transform'])
        # ROI sur la tf
        self.lr = pg.LinearRegionItem([0, 1])
        self.lr.setZValue(-10)
        self.tf_viewer.viewer.plotwidget.addItem(self.lr)

        # Plot de la phase
        self.dock_phase = Dock('Phase')
        widget_phase = QtWidgets.QWidget()
        self.phase_viewer = Viewer0D(widget_phase)
        self.dock_phase.addWidget(widget_phase)
        self.pid_controller.dock_area.addDock(self.dock_phase, 'bottom', self.dock_tf)
        self.phase_viewer.show_data([np.random.normal(size=100)])

        # add unit
        self.pid_controller.toolbar_layout.itemAt(8).widget().setText(
            'Current Value (fs): ' if self.settings.child('unitsGroup', 'convertFemto').value() else 'Current Value (deg): ')
        self.pid_controller.toolbar_layout.itemAt(7).widget().setText(
            'Target Value (fs): ' if self.settings.child('unitsGroup',
                                                          'convertFemto').value() else 'Target Value (deg): ')

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == 'show_camera':
            if not param.value():
                self.d1.hide()
            else:
                self.d1.show()
        if param.name() == 'show_roi':
            if not param.value():
                self.u.hide()
            else:
                self.u.show()
        if param.name() == 'show_fft':
            if not param.value():
                self.r.hide()
            else:
                self.r.show()
        if param.name() == 'show_phase':
            if not param.value():
                self.ph.hide()
            else:
                self.ph.show()
        if param.name() == 'wavelength':
            self.wavelength = param.value()
            print('wavelength changed to: ', self.wavelength)
        if param.name() == 'setDelayToZero':
            self.offset = self.phi
            print('The offset phase is now: ', self.offset)
        if param.name() == 'record':
            if (self.h5saver.h5_file_name is None) | (self.h5saver.h5_file_name == ''):
                if self.init_saver():   # File was initialized correctly
                    self.start_saver()
            else:
                self.start_saver()
        if param.name() == 'recordStop':
            self.close_saver()

        if param.name() == 'convertFemto':
            self.pid_controller.toolbar_layout.itemAt(8).widget().setText(
                'Current Value (fs): ' if param.value() else 'Current Value (deg): ')
            self.pid_controller.toolbar_layout.itemAt(7).widget().setText(
                'Target Value (fs): ' if param.value() else 'Target Value (deg): ')
            self.pid_controller.input_viewer.update_labels(
                'Input (fs)' if param.value() else 'Input (deg)')

    def init_saver(self):
           # First time we click on the button
            _, _, dataset_path = self.h5saver.update_file_paths()
            self.h5saver.settings.child('base_name').setValue(os.path.split(dataset_path)[0])
            try:
                self.h5saver.init_file(custom_naming=True)
            except AttributeError:
                logger.error('User didn\'t select a file')

            if not self.h5saver.h5_file_name == '':     # User selected a file
                # Scan
                self.h5saver.add_scan_group(title='PID Log')
                self.current_scan_path = self.h5saver.settings.child('current_scan_path').value()
                # Detector
                self.h5saver.add_det_group(
                    where=self.h5saver.current_scan_group,
                    title='Phase')
                phase_group = self.h5saver.current_group
                # Data
                self.h5saver.add_data_group(
                    where=phase_group,
                    group_data_type='data0D',
                    title='Phase'
                )
                phase_data_group = self.h5saver.current_group

                # Phase
                self.h5saver.add_CH_group(
                    where=phase_data_group,
                    title='Phase'
                )
                # Time
                self.h5saver.add_CH_group(
                    where=phase_data_group,
                    title='Time'
                )
                return True
            else:   # User didn't select a file
                # self.h5saver.h5_file_name = None
                return False

    def start_saver(self):
        self.recording = True
        self.settings.child('statsGroup', 'record').setOpts(visible=False)
        self.settings.child('statsGroup', 'recordStop').setOpts(visible=True)

        # self.timearray = np.zeros((self.settings.child('statsGroup', 'N_record').value(), 3))
        # self.delayarray = np.zeros(self.settings.child('statsGroup', 'N_record').value())
        # self.Nrecorded = 0
        # self.recording = True
        # self.h5saver = H5Saver()
        # time_init = datetime.now()
        # self.h5saver.init_file(custom_naming=False,
        #                         addhoc_file_path='C:/Users/mguer/Matthieu/recording_{0.year}{0.month}{0.day}_{0.hour}h{0.minute}.h5'.format(
        #                             time_init))
        #
        # self.h5saver.add_data_group('/', title='Time', group_data_type='data1D')
        # self.h5_timegroup = self.h5saver.current_group
        # self.h5saver.add_data_group('/', title='Delay', group_data_type='data1D')
        # self.h5_delaygroup = self.h5saver.current_group
        #
        self.settings.child('statsGroup', 'record').setOpts(visible=False)
        self.settings.child('statsGroup', 'recordStop').setOpts(visible=True)
        # print('saver_inited')

    def close_saver(self):
        self.recording = False
        self.settings.child('statsGroup', 'record').setOpts(visible=True)
        self.settings.child('statsGroup', 'recordStop').setOpts(visible=False)
        # self.h5saver.close()

    def ini_model(self):
        super().ini_model()

        self.phase_vector = deque(maxlen=100)
        self.offset = 0

    def convert_input(self, measurements):
        """
        Convert the image of the camera into x and y positions of the center of the beam.
        Parameters
        ----------
        measurements: (Ordereddict) Data from the camera

        Returns
        -------
        tuple: the coordinate of the center of the beam
        """
        key = list(measurements[self.detectors_name[0]]['data2D'].keys())[
            0]  # so it can also be used from another plugin having another key
        image = np.array(measurements[self.detectors_name[0]]['data2D'][key]['data'])


        # self.img1b.setImage(self.roi.getArrayRegion(image, self.img1a), levels=(0, image.max()))
        roi = self.camera_viewer.ROIselect.getArrayRegion(image,
                                                          self.camera_viewer.view.data_displayer.get_image('red'))

        self.fringes = np.nanmean(roi, axis=0)
        first_nonzero = (self.fringes!=0).argmax()
        last_nonzero = len(self.fringes) - np.flip(self.fringes != 0).argmax()
        self.fringes = self.fringes[first_nonzero:last_nonzero]
        self.fringes -= np.nanmean(self.fringes)

        tf = np.fft.rfft(np.fft.fftshift(self.fringes))
        # S = ft(self.fringes)

        x_min = int(self.lr.getRegion()[0])
        x_max = int(self.lr.getRegion()[1])
        # phase_roi = np.unwrap(np.angle(S[x_min:x_max]))

        phiwrapped = np.mean(np.angle(tf)[x_min:x_max])
        phi = np.unwrap(np.concatenate((self.phase_vector, [phiwrapped])))[-1]
        self.phase_vector.append(phi)
        self.phi = phi
        if self.settings.child('show_plots', 'show_camera').value():
            self.camera_viewer.show_data(DataFromPlugins(data=[image]))
        if self.settings.child('show_plots', 'show_roi').value():
            self.fringe_viewer.show_data([self.fringes])
        if self.settings.child('show_plots', 'show_fft').value():
            self.tf_viewer.show_data([np.abs(tf)])
        if self.settings.child('show_plots', 'show_phase').value():
            tmp = np.angle(np.exp(np.array(self.phase_vector) * 1j))
            self.phase_viewer.show_data([[tmp[-1]]])

        rms = np.std(self.phase_vector)

        if self.settings.child('unitsGroup', 'convertFemto').value():
            delay = (self.phi - self.offset) * self.wavelength * 1e-9 / (2 * np.pi * 3e8) * 1e15
            rms *= self.wavelength * 1e-9 / (2 * np.pi * 3e8) * 1e15
        else:
            delay = (self.phi - self.offset) / np.pi * 180
            rms *= np.pi/180

        self.settings.child('statsGroup', 'RMS').setValue(rms)

        if self.recording:
            if self.phase_arrays is None:
                self.phase_arrays = self.h5saver.add_data(
                    channel_group=self.h5saver.get_set_group(
                        where=self.h5saver.current_scan_group.path + '/Detector000/Data0D',
                        name='Ch000'),
                    data_dict=dict(data=np.asarray([delay])),
                    title='Current phase or delay',
                    enlargeable=True)

                self.time_axis_arrays = self.h5saver.add_data(
                    channel_group=self.h5saver.get_set_group(
                        where=self.h5saver.current_scan_group.path + '/Detector000/Data0D',
                        name='Ch001'),
                    data_dict=dict(data=np.asarray(time.mktime(datetime.now().timetuple()))),
                    title='Time since epoch',
                    enlargeable=True)

                self.h5saver.h5_file.flush()

            else:
                self.phase_arrays.append(np.asarray(delay))
                self.time_axis_arrays.append(np.asarray(time.mktime(datetime.now().timetuple())))
                self.h5saver.h5_file.flush()

        return InputFromDetector([delay])

    def convert_output(self, outputs, dt=0, stab=True):
        """
        Convert the output of the PID in units to be fed into the actuator
        Parameters
        ----------
        output: (float) output value from the PID from which the model extract
         a value of the same units as the actuator

        Returns
        -------
        list: the converted output as a list (if there are a few actuators)

        """

        if None in outputs:
            self.curr_output = np.zeros(np.shape(outputs))
        else:
            if self.settings.child('unitsGroup', 'convertFemto').value():
                self.curr_output = np.array(outputs) * 3e8 * 1e-15 / self.settings.child('unitsGroup',
                                                                                         'actUnits').value() / 2
            else:
                self.curr_output = np.array(outputs) / 360 * self.wavelength * 1e-9 / 2 / self.settings.child(
                    'unitsGroup', 'actUnits').value()

        return OutputToActuator(mode='rel', values=self.curr_output)


def main():
    from pymodaq.dashboard import DashBoard
    from pymodaq.daq_utils.daq_utils import get_set_preset_path
    from pymodaq.daq_utils import gui_utils as gutils
    from pathlib import Path
    from PyQt5 import QtWidgets
    from pymodaq.pid.pid_controller import DAQ_PID

    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    area = gutils.DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')

    dashboard = DashBoard(area)
    file = Path(get_set_preset_path()).joinpath("mock_fringe_stabilization.xml")
    if file.exists():
        dashboard.set_preset_mode(file)
        # prog.load_scan_module()
        pid_area = gutils.DockArea()
        pid_window = QtWidgets.QMainWindow()
        pid_window.setCentralWidget(pid_area)

        prog = DAQ_PID(pid_area)
        prog.set_module_manager(dashboard.detector_modules, dashboard.actuators_modules)
        QtWidgets.QApplication.processEvents()


    else:
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(f"The default file specified in the configuration file does not exists!\n"
                       f"{file}\n"
                       f"Impossible to load the DAQ_PID Module")
        msgBox.setStandardButtons(msgBox.Ok)
        ret = msgBox.exec()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



