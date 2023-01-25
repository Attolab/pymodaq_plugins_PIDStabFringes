from qtpy import QtWidgets
from qtpy.QtCore import Signal, QThread
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo, DataFromPlugins
import numpy as np
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base
from easydict import EasyDict as edict
from collections import OrderedDict
from pymodaq.daq_utils.daq_utils import gauss1D
from pymodaq.control_modules.viewer_utility_classes import comon_parameters
from pymodaq_plugins_pid.hardware.boiler import BoilerController

class DAQ_0DViewer_Boiler(DAQ_Viewer_base):
    """
        =============== =================
        **Attributes**  **Type**
        *params*        dictionnary list
        *x_axis*        1D numpy array
        *ind_data*      int
        =============== =================
    """
    params = comon_parameters +\
             [{'title:': 'Noise', 'name': 'noise', 'type': 'float', 'value': BoilerController._noise},
              {'title:': 'Ambiant temp', 'name': 'ambiant_temp', 'type': 'float',
               'value': BoilerController._ambiant_temperature}
              ]


    def __init__(self, parent=None,
                 params_state=None):  # init_params is a list of tuple where each tuple contains info on a 1D channel (Ntps,amplitude, width, position and noise)
        super().__init__(parent, params_state)
        self.ind_data = 0

    def commit_settings(self, param):
        """
            Setting the mock data.

            ============== ========= =================
            **Parameters**  **Type**  **Description**
            *param*         none      not used
            ============== ========= =================

            See Also
            --------
            set_Mock_data
        """
        if param.name() == 'noise':
            self.controller.noise = param.value()
        elif param.name() == 'ambiant_temp':
            self.controller.ambiant_temp = param.value()


    def ini_detector(self, controller=None):
        """
            Initialisation procedure of the detector.

            Returns
            -------
            ???
                the initialized status.

            See Also
            --------
            set_Mock_data
        """

        self.status.update(edict(initialized=False, info="", x_axis=None, y_axis=None, controller=None))
        if self.settings.child(('controller_status')).value() == "Slave":
            if controller is None:
                raise Exception('no controller has been defined externally while this detector is a slave one')
            else:
                self.controller = controller
        else:
            self.controller = BoilerController()

        self.status.initialized = True
        self.status.controller = self.controller
        return self.status

    def close(self):
        """
            not implemented.
        """
        pass

    def grab_data(self, Naverage=1, **kwargs):
        """
            | Start new acquisition.

            For each data on data_mock :
                * shift right data of ind_data positions
                * if naverage parameter is defined append the mean of the current data to the data to be grabbed.

            | Send the data_grabed_signal once done.

            =============== ======== ===============================================
            **Parameters**  **Type**  **Description**
            *Naverage*      int       specify the threshold of the mean calculation
            =============== ======== ===============================================

        """
        temperature = self.controller.grab()
        self.data_grabed_signal.emit([DataFromPlugins(name='Boiler', data=[np.array([temperature])],
                                                          dim='Data0D', labels=['Temperature'])])

