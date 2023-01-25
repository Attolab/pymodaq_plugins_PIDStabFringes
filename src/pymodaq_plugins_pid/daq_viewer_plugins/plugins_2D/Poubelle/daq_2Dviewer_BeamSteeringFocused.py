from qtpy.QtCore import QThread
from qtpy import QtWidgets
import numpy as np
import pymodaq.daq_utils.daq_utils as mylib
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, main
from easydict import EasyDict as edict
from collections import OrderedDict
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo, DataFromPlugins, Axis, my_moment
from pymodaq.control_modules.viewer_utility_classes import comon_parameters
from pymodaq_plugins_pid.hardware.beamsteering import BeamSteeringController
from scipy.ndimage.measurements import center_of_mass

class DAQ_2DViewer_BeamSteeringFocused(DAQ_Viewer_base):
    """
        =============== ==================
        **Attributes**   **Type**
        *params*         dictionnary list
        *x_axis*         1D numpy array
        *y_axis*         1D numpy array
        =============== ==================

        See Also
        --------
        utility_classes.DAQ_Viewer_base
    """

    params = comon_parameters

    def __init__(self, parent=None, params_state=None):
        # init_params is a list of tuple where each tuple contains info on a 1D channel (Ntps,amplitude,
        # width, position and noise)

        super().__init__(parent, params_state)
        self.x_axis = None
        self.y_axis = None
        self.live = False
        self.ind_commit = 0
        self.ind_data = 0

    def commit_settings(self, param):
        """
            Activate parameters changes on the hardware.

            =============== ================================ ===========================
            **Parameters**   **Type**                          **Description**
            *param*          instance of pyqtgraph Parameter   the parameter to activate
            =============== ================================ ===========================

            See Also
            --------
            set_Mock_data
        """
        pass

    def ini_detector(self, controller=None):
        """
            Initialisation procedure of the detector initializing the status dictionnary.

            See Also
            --------
            daq_utils.ThreadCommand, get_xaxis, get_yaxis
        """
        self.status.update(edict(initialized=False, info="", x_axis=None, y_axis=None, controller=None))
        try:

            if self.settings.child(('controller_status')).value() == "Slave":
                if controller is None:
                    raise Exception('no controller has been defined externally while this detector is a slave one')
                else:
                    self.controller = controller
            else:
                self.controller = BeamSteeringController()

            self.x_axis = self.controller.get_xaxis()
            self.y_axis = self.controller.get_yaxis()

            self.status.x_axis = self.x_axis
            self.status.y_axis = self.y_axis
            self.status.initialized = True
            self.status.controller = self.controller
            return self.status

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [getLineInfo() + str(e), 'log']))
            self.status.info = getLineInfo() + str(e)
            self.status.initialized = False
            return self.status

    def close(self):
        """
            not implemented.
        """
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """
            | For each integer step of naverage range set mock data.
            | Construct the data matrix and send the data_grabed_signal once done.

            =============== ======== ===============================================
            **Parameters**  **Type**  **Description**
            *Naverage*      int       The number of images to average.
                                      specify the threshold of the mean calculation
            =============== ======== ===============================================

            See Also
            --------
            set_Mock_data
        """

        image = self.controller.get_data_output(data_dim='2D')
        self.data_grabed_signal.emit([DataFromPlugins(name='Mock2DPID', data=[image**2], dim='Data2D')])

    def stop(self):
        return ""


if __name__ == '__main__':
    main(__file__)
