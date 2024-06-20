from typing import Union
import numpy as np
import uuid

class MeasurementRequest(object):
    """
    Indicates the existance of an event of interest at a given target point
    and requests agents to perform an observatrion with a given set of instruments

    ### Attributes:
        - requester (`str`): name of agent requesting the observations
        - target (`list`): location of the target area of interest expressed in [lat[deg], lon[deg], alt[km]]
        - severity (`float`): severity of the event being measured
        - observation_types (`list`): measurement types required to perform this task
        - t_start (`float`): start time of the availability of this task in [s] from the beginning of the simulation
        - t_end (`float`): end time of the availability of this task in [s] from the beginning of the simulation
        - t_corr (`float`): maximum decorralation time between different observations
        - id (`str`) : identifying number for this task in uuid format
    """        
    def __init__(self, 
                 requester : str,
                 target : list,
                 severity : float,
                 observations_types : list,
                 t_start: Union[float, int] = 0.0, 
                 t_end: Union[float, int] = np.Inf, 
                 t_corr: Union[float, int] = np.Inf, 
                 id: str = None, 
                 **_
                ) -> None:
        """
        Creates an instance of a measurement request 

        ### Arguments:
            - requester (`str`): name of agent requesting the observations
            - target (`list`): location of the target area of interest expressed in [lat[deg], lon[deg], alt[km]]
            - severity (`float`): severity of the event being measured
            - observation_types (`list`): measurement types of observations required to perform this task
            - t_start (`float`): start time of the availability of this task in [s] from the beginning of the simulation
            - t_end (`float`): end time of the availability of this task in [s] from the beginning of the simulation
            - t_corr (`float`): maximum decorralation time between different observations
            - id (`str`) : identifying number for this task in uuid format
        """
        # check arguments 
        if not isinstance(requester, str):
            raise ValueError(f'`rqst` must be of type `str`. Is of type {type(requester)}.')
        if not isinstance(target, list):
            raise ValueError(f'`target` must be of type `list`. is of type {type(target)}.')
        if any([not isinstance(target_val, float) and not isinstance(target_val, int) for target_val in target]):
            raise ValueError(f'`target` must be a `list` of elements of type `float` or type `int`.')
        if len(target) != 3:
            raise ValueError(f'`target` must be a list of size 3. Is of size {len(target)}.')
        if not isinstance(severity, float) and not isinstance(severity, int):
            raise ValueError(f'`severity` must be of type `float` or type `int`. is of type {type(severity)}.')
        if not isinstance(observations_types, list):
            raise ValueError(f'`instruments` must be of type `list`. is of type {type(observations_types)}.')
        if any([not isinstance(observations_type, str) for observations_type in observations_types]):
            raise ValueError(f'`measurements` must a `list` of elements of type `str`.')
        
        if isinstance(t_start, str) and t_start.lower() == "inf":   t_start = np.Inf
        if isinstance(t_end, str)   and t_end.lower() == "inf":     t_end = np.Inf
        if isinstance(t_corr, str)  and t_corr.lower() == "inf":    t_corr = np.Inf
        
        if t_start > t_end: raise ValueError(f"`t_start` must be lesser than `t_end`")
        if t_corr < 0:      raise ValueError(f"`t_corr` must have a non-negative value.")
        
        # initialize attributes
        self.requester = requester
        self.target = [target_val for target_val in target]
        self.severity = severity
        self.observations_types = [obs_type for obs_type in observations_types]    
        self.t_start = t_start
        self.t_end = t_end
        self.t_corr = t_corr
        self.id = str(uuid.UUID(id)) if id is not None else str(uuid.uuid1())
        
    def __repr__(self):
        task_id = self.id.split('-')
        return f'MeasurementReq_{task_id[0]}'

    def to_dict(self) -> dict:
        """
        Crates a dictionary containing all information contained in this measurement request object
        """
        return dict(self.__dict__)

    def from_dict(d : dict) -> object:
        return MeasurementRequest(**d)
    
    def __eq__(self, other : object) -> bool:
        if not isinstance(other, MeasurementRequest):
            raise ValueError(f'cannot compare `MeasurementRequest` object to an object of type {type(other)}.')
        
        my_dict : dict = self.to_dict()
        other_dict : dict = other.to_dict()

        my_dict.pop('id')
        other_dict.pop('id')

        return my_dict == other_dict
            
    def same_event(self, other : object) -> bool:
        """ compares the events being requested for observation between two measurement requests """

        if not isinstance(other, MeasurementRequest):
            raise ValueError(f'cannot compare `MeasurementRequest` object to an object of type {type(other)}.')

        same_target = all([abs(self.target[i]-other.target[i]) <= 1e-3 for i in range(len(self.target))])
        same_severity = abs(self.severity - other.severity) <= 1e-3
        same_observations = (len(self.observations_types) == len(other.observation_types)
                             and all([observation in other.observation_types for observation in self.observations_types]))
        same_time = abs(self.t_end - other.t_end) <= 1e-3
        same_decorrelation = abs(self.t_corr - other.t_corr) <= 1e-3
        return (same_target
                and same_severity
                # and same_observations
                and same_time
                and same_decorrelation
                )

    def __hash__(self) -> int:
        return hash(repr(self))

    def copy(self) -> object:
        return MeasurementRequest.from_dict(self.to_dict())
    
    # FOLLOWING 3 LINES WOULD GO WITHIN THE CONSTRUCTOR
    #     self.observation_groups = self.generate_observations_groups(observations_types)
    #     self.dependency_matrix = self.generate_dependency_matrix()
    #     self.time_dependency_matrix = self.generate_time_dependency_matrix()

    # def generate_observations_groups(self, observations_types : list) -> list:
    #     """
    #     Generates all combinations of groups of obvservations to be performed by a single or multiple agents

    #     ### Arguments:
    #         - observations_types (`list`): list of the observations that are needed to fully perform this task

    #     ### Returns:
    #         - observations_groups (`list`): list of observations group tuples containing the main observation type and a list of all dependent observations
    #     """
    #     # create measurement groups
    #     n_types = len(observations_types)
    #     observation_groups = []
    #     for r in range(1, n_types+1):
    #         combs = list(combinations(observations_types, r))
            
    #         for comb in combs:
    #             measurement_group = list(comb)
    #             main_measurement_permutations = list(permutations(comb, 1))

    #             for main_measurement in main_measurement_permutations:
    #                 main_measurement = list(main_measurement).pop()

    #                 dependend_measurements = copy.deepcopy(measurement_group)
    #                 dependend_measurements.remove(main_measurement)

    #                 if len(dependend_measurements) > 0:
    #                     observation_groups.append((main_measurement, dependend_measurements))
    #                 else:
    #                     observation_groups.append((main_measurement, []))
        
    #     return observation_groups     
    
    # def generate_dependency_matrix(self) -> list:
    #     # create dependency matrix
    #     dependency_matrix = []
    #     for index_a in range(len(self.observation_groups)):
    #         main_a, dependents_a = self.observation_groups[index_a]

    #         dependencies = []
    #         for index_b in range(len(self.observation_groups)):
    #             main_b, dependents_b = self.observation_groups[index_b]

    #             if index_a == index_b:
    #                 dependencies.append(0)

    #             elif main_a not in dependents_b or main_b not in dependents_a:
    #                 dependencies.append(-1)

    #             elif main_a == main_b:
    #                 dependencies.append(-1)
                    
    #             else:
    #                 dependents_a_extended : list = copy.deepcopy(dependents_a)
    #                 dependents_a_extended.remove(main_b)
    #                 dependents_b_extended : list = copy.deepcopy(dependents_b)
    #                 dependents_b_extended.remove(main_a)

    #                 if dependents_a_extended == dependents_b_extended:
    #                     dependencies.append(1)
    #                 else:
    #                     dependencies.append(-1)
            
    #         dependency_matrix.append(dependencies)
       
    #     return dependency_matrix

    # def generate_time_dependency_matrix(self) -> list:
    #     time_dependency_matrix = []

    #     for index_a in range(len(self.observation_groups)):
    #         time_dependencies = []
    #         for index_b in range(len(self.observation_groups)):
    #             if self.dependency_matrix[index_a][index_b] > 0:
    #                 time_dependencies.append(self.t_corr)
    #             else:
    #                 time_dependencies.append(numpy.Inf)
    #         time_dependency_matrix.append(time_dependencies)

    #     return time_dependency_matrix
