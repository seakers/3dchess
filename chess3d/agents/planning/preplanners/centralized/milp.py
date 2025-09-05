from collections import defaultdict
import logging
import os
import numpy as np

from dmas.utils import runtime_tracker
from dmas.modules import ClockConfig
from typing import Dict

import pandas as pd

from chess3d.agents.actions import ObservationAction
from chess3d.agents.planning.preplanners.centralized.dealer import DealerPreplanner
from chess3d.agents.planning.tasks import GenericObservationTask, SpecificObservationTask
from chess3d.agents.planning.tracker import ObservationHistory
from chess3d.agents.states import SimulationAgentState
from chess3d.mission.mission import Mission
from chess3d.orbitdata import OrbitData
from chess3d.utils import Interval


class DealerMILPPreplanner(DealerPreplanner):
    """
    A preplanner that generates plans for other agents using MILP models.
    """
    EARLIEST = 'earliest'
    BEST = 'best'

    def __init__(self, 
                 clients : Dict[str, OrbitData],
                 objective : str, 
                 model : str, 
                 licence_path : str = None, 
                 horizon : float = np.Inf,
                 period : float = np.Inf,
                 max_tasks : float = np.Inf,
                 debug : bool = False,
                 logger : logging.Logger = None):
        super().__init__(clients, horizon, period, debug, logger)

        if not debug or licence_path is not None:
            # Check for Gurobi license
            assert os.path.isfile(licence_path), f"Provided Gurobi licence path `{licence_path}` is not a valid file."

            # Set Gurobi license environment variable
            os.environ['GRB_LICENSE_FILE'] = licence_path

        # Validate inputs
        assert objective in ["reward", "duration"], "Objective must be either 'reward' or 'duration'."
        assert model in [self.EARLIEST, self.BEST], f"Model must be either '{self.EARLIEST}' or '{self.BEST}'."
        assert (isinstance(max_tasks, int) and max_tasks > 0) or max_tasks == np.Inf, "Max tasks must be a positive integer."

        # Set attributes
        self.objective = objective
        self.model = model
        self.max_tasks = max_tasks
        self.cross_track_fovs : Dict[str, Dict[str, float]] = self.__estimate_cross_track_fovs(clients)

    def __estimate_cross_track_fovs(self, clients : Dict[str, OrbitData]):
        cross_track_fovs : Dict[str, Dict[str, float]] = {client: defaultdict(float) for client in clients}
        for client, client_orbitdata in clients.items():
            # convert access data to dataframe
            data = {column : datum 
                    for column,datum in client_orbitdata.gp_access_data.data.items()}
            data['t'] = client_orbitdata.gp_access_data.t
            df = pd.DataFrame(data)

            # group observations by instrument name and time
            for group,grouped_data in df.groupby(['instrument', 't']):
                instrument = group[0]

                # estimate fov span as max - min look angle at a given time
                # NOTE this is a simplification and may not be accurate for all instruments
                fov_span = grouped_data['look angle [deg]'].max() - grouped_data['look angle [deg]'].min()

                # update cross-track fov if larger than previous estimate
                if instrument not in cross_track_fovs[client] or fov_span > cross_track_fovs[client][instrument]:
                    cross_track_fovs[client][instrument] = fov_span

        return cross_track_fovs

    @runtime_tracker
    def _generate_client_plans(self, 
                               state : SimulationAgentState, 
                               specs : object, 
                               clock_config : ClockConfig, 
                               orbitdata : OrbitData, 
                               mission : Mission, 
                               tasks : list, 
                               observation_history : ObservationHistory):
        """
        Generates plans for each agent based on the provided parameters.
        """
        # Outline planning horizon interval
        planning_horizon = Interval(state.t, state.t + self.horizon)

        # get only available tasks
        available_tasks : list[GenericObservationTask] = self.get_available_tasks(tasks, planning_horizon)
        
        # calculate coverage opportunities for tasks
        access_opportunities : Dict[str, int, int, str, tuple] = {client : self.calculate_access_opportunities(state, planning_horizon, client_orbitdata)
                                              for client, client_orbitdata in self.clients.items()
                                            }

        # create schedulable tasks from known tasks and future access opportunities
        schedulable_tasks : Dict[str, list[SpecificObservationTask]] = {client : self.create_tasks_from_accesses(available_tasks, client_access_opportunities, self.cross_track_fovs[client], orbitdata)
                                                                        for client, client_access_opportunities in access_opportunities.items()
                                                                        }
        
        raise NotImplementedError("MILP preplanner is not yet implemented.")