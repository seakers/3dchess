from abc import ABC, abstractmethod
import os
from instrupy.base import Instrument
import numpy as np
import pandas as pd
from chess3d.messages import *
from chess3d.agents.science.requests import MeasurementRequest

class DataProcessor(ABC):
    """
    # Data Processor
    Processes observation data and provides ev
    """
    def __init__(self, parent_name : str):
        super().__init__()
    
        self.parent_name = parent_name
        self.known_reqs = set()

    def process(self, senses : list) -> list:
        # unpack and sort incoming senses
        observations : list[ObservationResultsMessage] = [(Instrument.from_dict(msg.instrument), msg.observation_data) 
                                                          for msg in senses 
                                                          if isinstance(msg, ObservationResultsMessage)
                                                          and msg.src == self.parent_name]
        incoming_reqs : list[MeasurementRequest] = [MeasurementRequest.from_dict(msg.req) 
                                                    for msg in senses 
                                                    if isinstance(msg, MeasurementRequestMessage)
                                                    and msg.req['severity'] > 0.0]

        # update list of known requests
        self.known_reqs.update(incoming_reqs)

        # initiate requests generated
        requests : list[MeasurementRequest] = list()

        # process observations
        for instrument,observation_data in observations:
            for obs in observation_data:
                # process observation
                lat_event,lon_event,t_start,t_end,t_corr,severity,observations_required \
                    = self.process_observation(instrument, **obs)

                # no event in observation; skip
                if severity < 1e-3: continue

                # generate measurement request 
                measurement_req = MeasurementRequest(self.parent_name,
                                                    [lat_event,lon_event,0.0],
                                                    severity,
                                                    observations_required,
                                                    t_start, t_end, t_corr)
                
                # check if another request has already been made for this event
                if any([measurement_req.same_event(req) for req in self.known_reqs]):
                    # another request has been made for this same event; ignore
                    continue

                # update list of known requests
                self.known_reqs.add(measurement_req)

                # update list of generated requests 
                requests.append(measurement_req)

        # return list of generated requests
        return requests    

    @abstractmethod
    def process_observation(self, 
                            instrument : Instrument,
                            **kwargs
                            ) -> tuple:
        """ Processes incoming observation data and returns the characteristics of the event being detected if this exists """

    
class LookupProcessor(DataProcessor):
    def __init__(self, events_path : str, parent_name : str):
        """ 
        ## Lookuup Table Science Module

        Has prior knowledge of all of the events that will occur during the simulation.
        Compares incoming observations to a predefined list of events to determine whether an event has been observed.
        """
        super().__init__(parent_name)

        # load predefined events
        self.events : pd.DataFrame = self.load_events(events_path)

        # initialize empty list of detected events
        self.events_detected = set()

        # initialize update timer
        self.t_update = None

    def load_events(self, events_path : str) -> pd.DataFrame:

        if not os.path.isfile(events_path):
            raise ValueError('`events_path` must point to an existing file.')

        return pd.read_csv(events_path)
    
    def process_observation(self, 
                            instrument : Instrument,
                            t_img : float,
                            lat : float,
                            lon : float,
                            **_
                            ) -> tuple:
        
        # update list of events to ignore expired events
        if self.t_update is None or abs(self.t_update - t_img) > 100.0:
            self.events = self.events[self.events['start time [s]'] + self.events['duration [s]'] >= t_img]
            self.t_update = t_img

        # query known events
        if len(self.events.columns) <= 6:
            observed_events = [ (lat_event,lon_event,t_start,duration,severity,measurements)
                                for lat_event,lon_event,t_start,duration,severity,measurements in self.events.values
                                # same location as the observation
                                if abs(lat - lat_event) <= 1e-3
                                and abs(lon - lon_event) <= 1e-3
                                # availability during the time of observation
                                and t_start <= t_img <= t_start+duration
                                # event requires observations of the same type as the one performed
                                and instrument.name in measurements
                                # event has not been detected before
                                and (lat_event,lon_event,t_start,duration,severity,measurements) not in self.events_detected 
                                ]
        else:
            observed_events = [ (lat_event,lon_event,t_start,duration,severity,measurements)
                                for _,lat_event,lon_event,t_start,duration,severity,measurements in self.events.values
                                # same location as the observation
                                if abs(lat - lat_event) <= 1e-3
                                and abs(lon - lon_event) <= 1e-3
                                # availability during the time of observation
                                and t_start <= t_img <= t_start+duration
                                # event requires observations of the same type as the one performed
                                and instrument.name in measurements
                                # event has not been detected before
                                and (lat_event,lon_event,t_start,duration,severity,measurements) not in self.events_detected 
                                ]
        
        # return highest-severity event at that given target
        if observed_events:
            # sort by severity  
            observed_events.sort(key= lambda a: a[4])
            
            # get next highest severity event
            event = observed_events[-1]
            
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
            t_corr = t_end-t_img # TODO add scientific reason for this

            return lat_event,lon_event,t_img,t_end,t_corr,severity,observations_required

        return np.NaN,np.NaN,-1,-1,0.0,0.0,[]