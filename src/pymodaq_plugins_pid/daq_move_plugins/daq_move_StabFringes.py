from pymodaq.control_modules.move_utility_classes import DAQ_Move_base  # base class
from pymodaq.daq_move.utility_classes import comon_parameters  # common set of parameters for all actuators
from pymodaq.daq_utils.daq_utils import ThreadCommand, getLineInfo  # object used to send info back to the main thread
from easydict import EasyDict as edict  # type of dict
from pymodaq_plugins_pid.hardware.stabfringesmock import StabFringesController

'''
Mock actuator for stabilisation of the mock detector daq_2Dviewer_StabFringes.
'''


class DAQ_Move_StabFringes(DAQ_Move_base):
    """
        Wrapper object to access the Mock fonctionnalities, similar wrapper for all controllers.

        =============== ==============
        **Attributes**    **Type**
        *params*          dictionnary
        =============== ==============
    """
    _controller_units = 'whatever'
    is_multiaxes = True
    stage_names = StabFringesController.axis
    _epsilon = 1

    params = [  # elements to be added in order to control your custom stage

        {'title': 'MultiAxes:', 'name': 'multiaxes', 'type': 'group', 'visible': is_multiaxes, 'children': [
            {'title': 'is Multiaxes:', 'name': 'ismultiaxes', 'type': 'bool', 'value': is_multiaxes,
                'default': False},
            {'title': 'Status:', 'name': 'multi_status', 'type': 'list', 'value': 'Master',
                'limits': ['Master', 'Slave']},
            {'title': 'Axis:', 'name': 'axis', 'type': 'list', 'limits': stage_names},

        ]},
        {'title':'Saved Positions', 'name': 'savedPositions', 'type':'group', 'children': [
            {'title': 'Saved Positions', 'name': 'positionsList', 'type':'list', 'limits':[0.0]}, 
            {'title': 'Save Current', 'name': 'saveCurrPos', 'type':'bool_push', 'value':False}, 
            {'title': 'Reset', 'name': 'resetSavedPos', 'type':'bool_push', 'value':False}, 
        ]
        }] + comon_parameters

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)
        self.defaultLims = [float(i) for i in self.settings.child('savedPositions', 'positionsList').opts['limits']]
        self.settings.child('savedPositions', 'positionsList').setLimits(self.defaultLims)


    def check_position(self):
        """
            Get the current position from the hardware with scaling conversion.

            Returns
            -------
            float
                The position obtained after scaling conversion.

            See Also
            --------
            DAQ_Move_base.get_position_with_scaling, daq_utils.ThreadCommand
        """
        pos = self.controller.check_position(self.settings.child('multiaxes', 'axis').value())
        # print('Pos from controller is {}'.format(pos))
        # pos=self.get_position_with_scaling(pos)
        self.current_position = pos
        self.emit_status(ThreadCommand('check_position', [pos]))
        return pos

    def close(self):
        """
          not implemented.
        """
        pass

    # def commit_settings(self, param):
    #     """
    #         | Activate any parameter changes on the PI_GCS2 hardware.
    #         |
    #         | Called after a param_tree_changed signal from DAQ_Move_main.

    #     """

    #     pass

    def commit_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == 'positionsList':
            self.move_Abs(param.value())
        if param.name() == 'saveCurrPos':
            print(self.settings.child('savedPositions', 'positionsList').opts['limits'])
            self.settings.child('savedPositions', 'positionsList').setLims(self.settings.child('savedPositions', 'positionsList').opts['limits'] + [self.current_position])
        if param.name() == 'resetSavedPos':
            self.settings.child('savedPositions', 'positionsList').setLimits(self.defaultLims)
            print(self.settings.child('savedPositions', 'positionsList').opts['limits'])


    def ini_stage(self, controller=None):
        """
            Initialize the controller and stages (axes) with given parameters.

            ============== ================================================ ==========================================================================================
            **Parameters**  **Type**                                         **Description**

            *controller*    instance of the specific controller object       If defined this hardware will use it and will not initialize its own controller instance
            ============== ================================================ ==========================================================================================

            Returns
            -------
            Easydict
                dictionnary containing keys:
                 * *info* : string displaying various info
                 * *controller*: instance of the controller object in order to control other axes without the need to init the same controller twice
                 * *stage*: instance of the stage (axis or whatever) object
                 * *initialized*: boolean indicating if initialization has been done corretly

            See Also
            --------
             daq_utils.ThreadCommand
        """
        try:
            # initialize the stage and its controller status
            # controller is an object that may be passed to other instances of DAQ_Move_Mock in case
            # of one controller controlling multiaxes

            self.status.update(edict(info="", controller=None, initialized=False))

            # check whether this stage is controlled by a multiaxe controller (to be defined for each plugin)

            # if mutliaxes then init the controller here if Master state otherwise use external controller
            if self.settings.child('multiaxes', 'ismultiaxes').value() and self.settings.child('multiaxes',
                                                                                               'multi_status').value() == "Slave":
                if controller is None:
                    raise Exception('no controller has been defined externally while this axe is a slave one')
                else:
                    self.controller = controller
            else:  # Master stage
                self.controller = StabFringesController()  # any object that will control the stages

            info = "Mock PID stage"
            self.status.info = info
            self.status.controller = self.controller
            self.status.initialized = True
            return self.status

        except Exception as e:
            self.emit_status(ThreadCommand('Update_Status', [getLineInfo() + str(e), 'log']))
            self.status.info = getLineInfo() + str(e)
            self.status.initialized = False
            return self.status

    def move_Abs(self, position):
        """
            Make the absolute move from the given position after thread command signal was received in DAQ_Move_main.

            =============== ========= =======================
            **Parameters**  **Type**   **Description**

            *position*       float     The absolute position
            =============== ========= =======================

            See Also
            --------
            DAQ_Move_base.set_position_with_scaling, DAQ_Move_base.poll_moving

        """
        position = self.check_bound(position)
        # position=self.set_position_with_scaling(position)
        # print(position)
        self.target_position = position
        self.controller.move_abs(self.target_position, self.settings.child('multiaxes', 'axis').value())


    def move_Rel(self, position):
        """
            Make the relative move from the given position after thread command signal was received in DAQ_Move_main.

            =============== ========= =======================
            **Parameters**  **Type**   **Description**

            *position*       float     The absolute position
            =============== ========= =======================

            See Also
            --------
            hardware.set_position_with_scaling, DAQ_Move_base.poll_moving

        """
        position = self.check_bound(self.current_position + position) - self.current_position
        self.target_position = position + self.current_position

        self.controller.move_rel(position, self.settings.child('multiaxes', 'axis').value())


    def move_Home(self):
        """
          Send the update status thread command.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        self.emit_status(ThreadCommand('Update_Status', ['Move Home not implemented']))

    def stop_motion(self):
        """
          Call the specific move_done function (depending on the hardware).

          See Also
          --------
          move_done
        """
        self.move_done()
