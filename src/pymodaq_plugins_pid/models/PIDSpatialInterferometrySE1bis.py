from PyQt5 import QtWidgets
import numpy as np
from pyqtgraph.dockarea import Dock
from collections import deque
from datetime import datetime
import os, time
import logging
from typing import List

import pyqtgraph as pg

from pymodaq.extensions.pid.utils import PIDModelGeneric, DataToActuatorPID
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.h5modules.saving import H5Saver
from pymodaq.utils.parameter import utils as putils
from pymodaq.utils.data import DataToExport, DataActuator, DataCalculated, DataFromPlugins
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import Axis

logger = set_logger(get_module_name(__file__))
pid_logger = logging.getLogger('pymodaq.pid_controller')

class PIDModelSpatialInterferometrySE1bis(PIDModelGeneric):
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
        {'title': 'Units', 'name': 'unitsGroup', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'Convert to femto', 'name': 'convertFemto', 'type': 'bool', 'value': True},
             {'title': 'Actuator unit (m)', 'name': 'actUnits', 'type': 'float', 'value': -1e-6},
             {'title': 'Set delay to zero', 'name': 'setDelayToZero', 'type': 'bool_push', 'value': False}
         ]},
        {'title': 'Stats', 'name': 'statsGroup', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'RMS', 'name': 'RMS', 'type': 'float', 'value': 0.0, 'readonly': True},
         ]},

        {'title': 'FFT peak settings', 'name': 'spectrum', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
             {'title': 'First index', 'name': 'omega_min', 'type': 'int', 'value': 0},
             {'title': 'Last index', 'name': 'omega_max', 'type': 'int', 'value': 1},
             {'title': 'Peak threshold', 'name': 'peak_threshold', 'type': 'float', 'value': 0.5}]},
        {'title': 'Camera ROI', 'name': 'roi', 'type': 'group', 'expanded': False, 'visible': True,
         'children': [
             {'title': 'x0', 'name': 'roi_x0', 'type': 'int', 'value': 0},
             {'title': 'y0', 'name': 'roi_y0', 'type': 'int', 'value': 0},
             {'title': 'width', 'name': 'roi_width', 'type': 'int', 'value': 100},
             {'title': 'height', 'name': 'roi_height', 'type': 'int', 'value': 100}]},
        {'title': 'Show plots', 'name': 'show_plots', 'type': 'group', 'expanded': False, 'visible': True,
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
        # self.camera_viewer = Viewer2D(widget_camera)
        self.dock_camera.addWidget(widget_camera)
        self.pid_controller.dock_area.addDock(self.dock_camera)

        self.img1a = pg.ImageItem()
        self.img1a.setImage(np.random.normal(size = (self.settings['roi','roi_x0']+self.settings['roi','roi_width'],self.settings['roi','roi_y0']+self.settings['roi','roi_height'])))
        w1 = pg.PlotWidget(title = "Camera Plot")¶
        w1.addItem(self.img1a)
        self.dock_camera.addWidget(w1)
        #add ROI
        self.roi = pg.RectROI([self.settings['roi','roi_x0'],self.settings['roi','roi_y0']],[self.settings['roi','roi_width'],self.settings['roi','roi_height']],pen=(5,30))
        self.roi.sigRegionChangeFinished.connect(self.update_camera_roi)
        self.roi.addRotateHandle([1,0],[0.5,0.5])
        w1.addItem(self.roi)
        # self.camera_viewer.show_data(DataFromPlugins(data=[np.random.normal(size=(100, 100))], labels='Camera Image', name='Camera Image'))
        # self.camera_viewer.view.get_action('ROIselect').trigger()
        # self.camera_viewer.view.ROIselect.setPen(pen=(5, 30))
        # self.camera_viewer.view.ROIselect.setSize([self.settings.child('roi','roi_width').value(), self.settings.child('roi','roi_height').value()])
        # self.camera_viewer.view.ROIselect.setPos([self.settings.child('roi', 'roi_x0').value(), self.settings.child('roi', 'roi_y0').value()])
        # self.camera_viewer.view.ROIselect.sigRegionChangeFinished.connect(self.update_camera_roi)


        # Plot de la TF des franges
        self.dock_tf = Dock('Fourier transform')
        u = pg.PlotWidget(title="Fourier transform")
        self.tf_viewer = u.plot()
        self.dock_tf.addWidget(u)
        self.pid_controller.dock_area.addDock(self.dock_tf, 'right', self.dock_camera)
        self.tf_viewer.setData(y = np.random.normal(size=100))

        # ROI sur la tf
        self.lr = pg.LinearRegionItem([self.settings.child('spectrum', 'omega_min').value(), self.settings.child('spectrum', 'omega_max').value()])
        self.lr.setZValue(-10)
        u.addItem(self.lr)
        self.lr.sigRegionChangeFinished.connect(self.update_tf_roi)

        # # Line for threshold
        self.tf_threshold = pg.InfiniteLine(self.settings.child('spectrum', 'peak_threshold').value(), angle=0, movable=True)
        u.addItem(self.tf_threshold)
        self.tf_threshold.sigPositionChangeFinished.connect(self.update_tf_threshold)

        # Plot des franges
        self.dock_fringes = Dock('Fringes')
        u = pg.PlotWidget(title="Fringes")
        # widget_fringes = QtWidgets.QWidget()
        self.fringe_viewer = u.plot()#Viewer1D(widget_fringes)
        self.dock_fringes.addWidget(u)
        self.pid_controller.dock_area.addDock(self.dock_fringes, 'bottom', self.dock_camera)
        self.fringe_viewer.setData(y = np.random.normal(size=100))
        logger.info('Init Widget: Dock ROI')

        # Plot de la phase
        self.dock_phase = Dock('Phase')
        p = pg.PlotWidget(title="Phase")
        self.phase_viewer = p.plot()
        self.dock_phase.addWidget(p)
        self.pid_controller.dock_area.addDock(self.dock_phase, 'bottom', self.dock_tf)
        self.phase_viewer.setData(y=np.random.normal(size = 100))

        # add unit
        if self.settings.child('unitsGroup', 'convertFemto').value():
            self.currlabel = QtWidgets.QLabel('Current Value (fs): ')
            self.setlabel = QtWidgets.QLabel('Target Value (fs): ')

        else:
            self.currlabel = QtWidgets.QLabel('Current Value (deg): ')
            self.setlabel = QtWidgets.QLabel('Target Value (deg): ')

        self.pid_controller.toolbar_layout.addWidget(self.currlabel, 4, 1, 1, 1)
        self.pid_controller.toolbar_layout.addWidget(self.setlabel , 3, 1, 1, 1)

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

        if param.name() == 'convertFemto':
            if self.settings.child('unitsGroup', 'convertFemto').value():
                label1 = 'Current Value (fs): '
                label2 = 'Target Value (fs): '
            else:
                label1 = 'Current Value (deg): '
                label2 = 'Target Value (deg): '
            self.currlabel.setText(label1)
            self.setlabel.setText(label2)

        if param.name() == 'omega_min' or 'omega_max':
            self.lr.sigRegionChangeFinished.disconnect(self.update_tf_roi)
            self.lr.setRegion([self.settings.child('spectrum', 'omega_min').value(),
                                           self.settings.child('spectrum', 'omega_max').value()])
            self.lr.sigRegionChangeFinished.connect(self.update_tf_roi)

        if param.name() == 'peak_threshold':
            self.tf_threshold.sigPositionChangeFinished.disconnect(self.update_tf_threshold)
            self.tf_threshold.setPos(param.value())
            self.tf_threshold.sigPositionChangeFinished.connect(self.update_tf_threshold)

        # if param.name() in putils.iter_children(
        #             self.settings.child("roi"), []):
        #     self.camera_viewer.view.ROIselect.sigRegionChangeFinished.disconnect(self.update_camera_roi)
        #     self.camera_viewer.view.ROIselect.setSize(
        #         [self.settings.child('roi', 'roi_width').value(), self.settings.child('roi', 'roi_height').value()])
        #     self.camera_viewer.view.ROIselect.setPos(
        #         [self.settings.child('roi', 'roi_x0').value(), self.settings.child('roi', 'roi_y0').value()])
        #     self.camera_viewer.view.ROIselect.sigRegionChangeFinished.connect(self.update_camera_roi)

    def update_tf_roi(self, linear_roi):
        pos = linear_roi.getRegion()
        self.settings.child("spectrum", "omega_min").setValue(
            round(pos[0])
        )
        self.settings.child("spectrum", "omega_max").setValue(
            round(pos[1])
        )
        linear_roi.setRegion((round(pos[0]),round(pos[1])))

    def update_tf_threshold(self, line):
        pos = line.value()
        self.settings.child("spectrum", "peak_threshold").setValue(
            pos
        )

    def update_camera_roi(self, roi):
        (x,y) = roi.pos()
        (w,h) = roi.size()
        self.settings.child('roi', 'roi_width').setValue(w)
        self.settings.child('roi', 'roi_height').setValue(h)
        self.settings.child('roi', 'roi_x0').setValue(x)
        self.settings.child('roi', 'roi_y0').setValue(y)


    def ini_model(self):
        super().ini_model()

        self.phase_vector = deque(maxlen=100)
        self.offset = 0

    def convert_input(self, measurements: DataToExport) -> DataToExport:
        """
        Convert the image of the camera into x and y positions of the center of the beam.
        Parameters
        ----------
        measurements: (Ordereddict) Data from the camera

        Returns
        -------
        tuple: the coordinate of the center of the beam
        """
        image = np.array(measurements.get_data_from_dim('Data2D')[0][0])

        #Sum on ROI to have fringes
        self.fringes = np.nanmean(self.roi.getArrayRegion(image, self.img1a),axis = 0)

        # self.img1b.setImage(self.roi.getArrayRegion(image, self.img1a), levels=(0, image.max()))
        # roi = self.camera_viewer.view.ROIselect.getArrayRegion(image,
        #                                                   self.camera_viewer.view.data_displayer.get_image('red'))

        # self.fringes = np.nanmean(roi, axis=0)
        first_nonzero = (self.fringes!=0).argmax()
        last_nonzero = len(self.fringes) - np.flip(self.fringes != 0).argmax()
        self.fringes = self.fringes[first_nonzero:last_nonzero]
        self.fringes -= np.nanmean(self.fringes)

        tf = np.fft.rfft(np.fft.fftshift(self.fringes))
        # S = ft(self.fringes)

        x_min = self.settings.child("spectrum", "omega_min").value()
        x_max = self.settings.child("spectrum", "omega_max").value()
        # phase_roi = np.unwrap(np.angle(S[x_min:x_max]))

        if len(tf[x_min:x_max]) != 0:   # Check that ROI is not empty
            phiwrapped = np.mean(np.angle(tf)[x_min:x_max])
        else:
            phiwrapped =0

        # Check that FFT peak is sufficiently strong
        if np.mean(np.abs(tf[x_min:x_max])) > self.settings.child('spectrum', 'peak_threshold'). value():
            # Restart PID in case it was stopped by fft peak too low
            if not self.pid_controller.is_action_checked('pause'):
                pid_logger.disabled = True #avoid having hundreds of messages in log
                self.pid_controller.command_pid.emit(ThreadCommand('pause_PID', [False]))
                pid_logger.disabled = False

            new_phase_point = True
            phi = np.unwrap(np.concatenate((self.phase_vector, [phiwrapped])))[-1]
            self.phase_vector.append(phi)
            self.phi = phi
            rms = np.std(self.phase_vector)

            if self.settings.child('unitsGroup', 'convertFemto').value():
                delay = (self.phi - self.offset) * self.wavelength * 1e-9 / (2 * np.pi * 3e8) * 1e15
                rms *= self.wavelength * 1e-9 / (2 * np.pi * 3e8) * 1e15
            else:
                delay = (self.phi - self.offset) / np.pi * 180
                rms *= np.pi / 180

            self.settings.child('statsGroup', 'RMS').setValue(rms)


        else:   # if FFT peak is too weak we just return the set point so that the PID does nothing
            # Force the PID to do nothing
            pid_logger.disabled = True  # avoid having hundreds of messages in log
            self.pid_controller.command_pid.emit(ThreadCommand('pause_PID', [True]))
            pid_logger.disabled = False

            new_phase_point = False
            delay = self.pid_controller.setpoints[0]

        if self.settings.child('show_plots', 'show_camera').value():
            self.img1a.setImage(image, axisOrder='row-major')
        if self.settings.child('show_plots', 'show_roi').value():
            self.fringe_viewer.setData(y=self.fringes)
        if self.settings.child('show_plots', 'show_fft').value():
            self.tf_viewer.setData(y=np.abs(tf))
        if self.settings.child('show_plots', 'show_phase').value():
            if new_phase_point:
                tmp = np.angle(np.exp(np.array(self.phase_vector) * 1j))
                self.phase_viewer.setData(tmp)
        #
        # if self.recording:
        #     if self.phase_arrays is None:
        #         self.phase_arrays = self.h5saver.add_data(
        #             channel_group=self.h5saver.get_set_group(
        #                 where=self.h5saver.current_scan_group.path + '/Detector000/Data0D',
        #                 name='Ch000'),
        #             data_dict=dict(data=np.asarray([delay])),
        #             title='Current phase or delay',
        #             enlargeable=True)
        #
        #         self.time_axis_arrays = self.h5saver.add_data(
        #             channel_group=self.h5saver.get_set_group(
        #                 where=self.h5saver.current_scan_group.path + '/Detector000/Data0D',
        #                 name='Ch001'),
        #             data_dict=dict(data=np.asarray(time.mktime(datetime.now().timetuple()))),
        #             title='Time since epoch',
        #             enlargeable=True)
        #
        #         self.h5saver.h5_file.flush()
        #
        #     else:
        #         self.phase_arrays.append(np.asarray(delay))
        #         self.time_axis_arrays.append(np.asarray(time.mktime(datetime.now().timetuple())))
        #         self.h5saver.h5_file.flush()

        self.curr_input = [delay]
        return DataToExport('inputs',
                            data=[DataCalculated(self.setpoints_names[ind],
                                                 data=[np.array([self.curr_input[ind]])])
                                  for ind in range(len(self.curr_input))])

    def convert_output(self, outputs: List[float], dt, stab=True) -> DataToActuatorPID:
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

        return DataToActuatorPID('pid', mode='rel',
                                 data=[DataActuator(self.actuators_name[0], data=[self.curr_output])])


def main():
    from pymodaq.dashboard import DashBoard
    from pymodaq.utils.daq_utils import get_set_preset_path
    from pymodaq.utils import gui_utils as gutils
    from pathlib import Path
    from PyQt5 import QtWidgets
    from pymodaq.extensions.pid.pid_controller import DAQ_PID

    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    area = gutils.DockArea()
    win.setCentralWidget(area)
    win.resize(1000, 500)
    win.setWindowTitle('PyMoDAQ Dashboard')

    dashboard = DashBoard(area)
    file = Path(get_set_preset_path()).joinpath("mock_stabfringes2.xml")
    if file.exists():
        dashboard.set_preset_mode(file)
        # prog.load_scan_module()
        pid_area = gutils.DockArea()
        pid_window = QtWidgets.QMainWindow()
        pid_window.setCentralWidget(pid_area)

        prog = DAQ_PID(pid_area, dashboard)
        #prog.set_module_manager(dashboard.detector_modules, dashboard.actuators_modules)
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



