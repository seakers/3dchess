from logging import Logger
import pandas as pd
from zmq import asyncio as azmq

from dmas.modules import NetworkConfig
from dmas.modules import *

from chess3d.agents.actions import ObservationAction
from chess3d.agents.states import SimulationAgentState
from chess3d.agents.science.requests import *
from chess3d.messages import *

from instrupy.base import Instrument

class ScienceModuleTypes(Enum):
    LOOKUP = 'LOOKUP'

class ScienceModule(InternalModule):
    def __init__(   self, 
                    results_path : str,
                    parent_name : str,
                    parent_network_config: NetworkConfig, 
                    logger: logging.Logger = None
                ) -> None:

        addresses = parent_network_config.get_internal_addresses()        
        sub_addesses = []
        sub_address : str = addresses.get(zmq.PUB)[0]
        sub_addesses.append( sub_address.replace('*', 'localhost') )

        pub_address : str = addresses.get(zmq.SUB)[1]
        pub_address = pub_address.replace('localhost', '*')

        addresses = parent_network_config.get_manager_addresses()
        push_address : str = addresses.get(zmq.PUSH)[0]

        science_network_config =  NetworkConfig(parent_name,
                                        manager_address_map = {
                                        zmq.REQ: [],
                                        zmq.SUB: sub_addesses,
                                        zmq.PUB: [pub_address],
                                        zmq.PUSH: [push_address]})

        super().__init__(   f"{parent_name}-SCIENCE_MODULE", 
                            science_network_config, 
                            parent_network_config, 
                            logging.INFO, 
                            logger)
        
        # initialize attributes
        self.events = None
        self.known_reqs = set()

        # assign parameters
        self.results_path = results_path
        self.parent_name = parent_name
    
    async def sim_wait(self, _: float) -> None:
        # is event-driven; no need to support timed delays
        return

    async def setup(self) -> None:
        # setup internal inboxes
        self.onboard_processing_inbox = asyncio.Queue()

    async def live(self) -> None:
        """
        Performs concurrent tasks:
        - Listener: receives messages from the parent agent and checks results
        - Science valuer: determines value of a given measurement
        - Science reasoning: checks data for outliers
        - Onboard processing: converts data of one type into data of another type (e.g. level 0 to level 1)
        """
        # announce existance to other modules 
        announcement_msg = SimulationMessage(self.get_element_name(), self.get_parent_name(), 'HANDSHAKE')
        await self._send_manager_msg(announcement_msg, zmq.PUB)

        # initialize concurrent tasks
        listener_task = asyncio.create_task(self.listener(), name='listener()')
        onboard_processing_task = asyncio.create_task(self.onboard_processing(), name='onboard_processing()')
        
        # wait for a task to terminate
        _, pending = await asyncio.wait([listener_task, onboard_processing_task], 
                                        return_when=asyncio.FIRST_COMPLETED)

        # cancel the remaning tasks
        for task in pending:
            task : asyncio.Task
            task.cancel()
            await task

    async def listener(self):
        """ Unpacks and classifies any incoming messages """
        # 
        try:
            # initiate results tracker
            results = {}

            # create poller for all broadcast sockets
            poller = azmq.Poller()
            manager_socket, _ = self._manager_socket_map.get(zmq.SUB)
            poller.register(manager_socket, zmq.POLLIN)

            # listen for broadcasts and place in the appropriate inboxes
            while True:
                self.log('listening to manager broadcast!')
                _, _, content = await self.listen_manager_broadcast()

                if content['msg_type'] == SimulationMessageTypes.SENSES.value:
                    self.log(f"received senses from parent agent!", level=logging.DEBUG)

                    # unpack message 
                    senses_msg : SensesMessage = SensesMessage(**content)

                    senses = []
                    senses.append(senses_msg.state)
                    senses.extend(senses_msg.senses)     

                    for sense in senses:
                        # unpack message
                        msg : SimulationMessage = message_from_dict(**sense)

                        # check type of message being received
                        if isinstance(msg, AgentStateMessage):
                            self.log(f"received agent state message!")
                                                        
                            # update current state
                            state : SimulationAgentState = SimulationAgentState.from_dict(msg.state)
                            await self.update_current_time(state.t)

                        elif isinstance(msg, ObservationResultsMessage):
                            # observation data from another agent was received
                            self.log(f"received observation data from agent!")

                            # send to onboard processor
                            await self.onboard_processing_inbox.put(msg)

                        elif isinstance(msg, MeasurementRequestMessage):
                            # measurement request message received
                            self.log(f"received measurement request message!")

                            # unapack measurement request
                            req : MeasurementRequest = MeasurementRequest.from_dict(msg.req)
                            
                            # update list of known measurement requests
                            if req.severity > 0.0: self.known_reqs.add(req)

                # if sim-end message, end agent `live()`
                elif content['msg_type'] == ManagerMessageTypes.SIM_END.value:
                    self.log(f"received manager broadcast of type {content['msg_type']}! terminating `live()`...",level=logging.INFO)
                    return

                elif content['msg_type'] == SimulationMessageTypes.OBSERVATION.value:
                    # unpack message
                    self.log(f"received manager broadcast of type {content['msg_type']}!",level=logging.WARN)
                    msg = ObservationResultsMessage(**content)
                    await self.onboard_processing_inbox.put(msg)
        
        except asyncio.CancelledError:
            print("Asyncio cancelled error in science module listener")
            return
                
    async def onboard_processing(self) -> None:
        """ 
        Processes incoming observation data and generates measurement requests if an event is detected.
        
        Always returns a measurement request regardless of the presense of an event following an observation.
        The severity is set depending on the whether there was an event detected or not.
        The planner module will only consider and broadcast measurement requests with a non-zero severity.
        
        """
        try:
            while True:
                # wait for next observation to be performed by parent agent
                msg : ObservationResultsMessage = await self.onboard_processing_inbox.get()

                # unpack information from observation message
                observation_data = msg.observation_data
                instrument = Instrument.from_dict(msg.instrument)
                
                # process each observation
                reqs_from_observations = []
                for obs in observation_data:
                    # process observation
                    lat_event,lon_event,t_start,t_end,t_corr,severity,observations_required \
                        = self.process_observation(instrument, **obs)

                    # generate measurement request 
                    measurement_req = MeasurementRequest(self.get_parent_name(),
                                                         [lat_event,lon_event,0.0],
                                                         severity,
                                                         observations_required,
                                                         t_start,
                                                         t_end,
                                                         t_corr)
                    
                    # check if another request has already been made for this event
                    if any([measurement_req.same_event(req) for req in self.known_reqs]):
                        # another request has been made for this same event; ignore
                        measurement_req.severity = 0.0
                    
                    if measurement_req.severity > 0.0:
                        x = 1
                    
                    # send request to all internal agent modules
                    req_msg = MeasurementRequestMessage(self.get_module_name(), 
                                                        self.get_parent_name(), 
                                                        measurement_req.to_dict())
                    reqs_from_observations.append(req_msg)
                
                if not reqs_from_observations:
                    measurement_req = MeasurementRequest(self.get_parent_name(),
                                                         [-1,-1,0.0],
                                                         0.0,
                                                         observations_required,
                                                         t_start,
                                                         t_end,
                                                         t_corr)
                    req_msg = MeasurementRequestMessage(self.get_module_name(), 
                                                        self.get_parent_name(), 
                                                        measurement_req.to_dict())
                    reqs_from_observations.append(req_msg)
                    
                for req_msg in reqs_from_observations:
                    await self._send_manager_msg(req_msg, zmq.PUB)


        except asyncio.CancelledError:
            return
        
    @abstractmethod
    def process_observation(self, 
                            instrument : Instrument,
                            **kwargs
                            ) -> tuple:
        """ Processes incoming observation data and returns the characteristics of the event being detected if this exists"""

    async def teardown(self) -> None:
        # nothing to tear-down
        return

class OracleScienceModule(ScienceModule):
    def __init__(self, 
                 results_path: str, 
                 events_path : str,
                 parent_name: str, 
                 parent_network_config: NetworkConfig, 
                 logger: Logger = None
                 ) -> None:
        """ 
        ## Oracle Science Module

        Has prior knowledge of all of the events that will occur during the simulation.
        Compares incoming observations to a predefined list of events to determine whether an event has been observed.
        """
        super().__init__(results_path, parent_name, parent_network_config, logger)

        # load predefined events
        self.events = pd.read_csv(events_path)

        # initialize empty list of detected events
        self.events_detected = set()
                
    def process_observation(self, 
                            instrument : Instrument,
                            t_img : float,
                            lat : float,
                            lon : float,
                            **_
                            ) -> tuple:
        
        # query known events
        observed_events = [ (lat_event,lon_event,t_start,duration,severity,measurements)
                            for lat_event,lon_event,t_start,duration,severity,measurements 
                            in self.events.values
                            if abs(lat - lat_event) <= 1e-3
                            and abs(lon - lon_event) <= 1e-3
                            and t_start <= t_img <= t_start+duration
                            and instrument.name in measurements
                            ]
        
        # sort by severity  
        observed_events.sort(key= lambda a: a[4])

        while observed_events:
            # get next highest severity event
            event = observed_events.pop()

            # check if event has already been detected before
            if event in self.events_detected:
                # already been detected, skip
                continue
            
            # add event to list of detected events
            self.events_detected.update(event)

            # unpackage event info
            lat_event,lon_event,t_start,duration,severity,observations_str = event 

            # get list of required observations
            observations_str : str = observations_str.replace('[','')
            observations_str : str = observations_str.replace(']','')
            observations_required : list = observations_str.split(',')
            observations_required.remove(instrument.name)

            # calculate end of event
            t_end = t_start+duration

            # estimate decorrelation time:
            t_corr = t_start+duration-t_img # TODO add scientific reason for this

            return lat_event,lon_event,t_img,t_end,t_corr,severity,observations_required

        return np.NaN,np.NaN,-1,-1,0.0,0.0,[]
    