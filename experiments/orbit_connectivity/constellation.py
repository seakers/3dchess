
from abc import ABC, abstractmethod
from copy import deepcopy
import json
import os
import shutil
from typing import Dict, List

import numpy as np
from orbitpy.mission import Mission

from chess3d.utils import print_welcome

class Constellation(ABC):
    R = 6378 # Earth radius in km    
    GM = 3.986004418e14 * (1/1000)**3 # [km^3/s^2]

    MISSION_TEMPLATE = {
                    "epoch": {
                        "@type": "GREGORIAN_UT1",
                        "year": 2020,
                        "month": 1,
                        "day": 1,
                        "hour": 0,
                        "minute": 0,
                        "second": 0
                    },
                    "duration": 0.75,
                    "propagator": {
                        "@type": "J2 ANALYTICAL PROPAGATOR",
                        # "stepSize": 10
                    },
                    "spacecraft": [],
                    "grid": [
                        {
                            "@type": "customGrid",
                            "covGridFilePath": "./grids/toy.csv"
                        }
                    ],
                    "scenario" : {
                        "connectivity": "LOS"
                    },
                    "settings": {
                        "coverageType": "GRID COVERAGE",
                        "outDir" : "./orbits"
                    }
                }
    
    SPACECRAFT_TEMPLATE = {
                        "@id": "sat_temp",
                        "name": "sat_temp",
                        "spacecraftBus": {
                            "name": "BlueCanyon",
                            "mass": 20,
                            "volume": 0.5,
                            "orientation": {
                                "referenceFrame": "NADIR_POINTING",
                                "convention": "REF_FRAME_ALIGNED"
                            }
                        },
                        "instrument": {
                            "name": "VNIR hyper",
                            "@id" : "vnir_hyp_imager",
                            "@type" : "VNIR",
                            "detectorWidth": 6.6e-6,
                            "focalLength": 3.6,  
                            "orientation": {
                                "referenceFrame": "NADIR_POINTING",
                                "convention": "REF_FRAME_ALIGNED"
                            },
                            "fieldOfViewGeometry": { 
                                "shape": "RECTANGULAR", 
                                "angleHeight": 2.5, 
                                "angleWidth": 2.5
                            },
                            "maneuver" : {
                                "maneuverType":"SINGLE_ROLL_ONLY",
                                "A_rollMin": -50,
                                "A_rollMax": 50
                            },
                            "spectral_resolution" : "Multispectral"
                        },
                        "orbitState": {
                            "date": {
                                "@type": "GREGORIAN_UT1",
                                "year": 2020,
                                "month": 1,
                                "day": 1,
                                "hour": 0,
                                "minute": 0,
                                "second": 0
                            },
                            "state": {
                                "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                                "sma": 7078, # ~700 km altitude
                                "ecc": 0.01,
                                "inc": 60.0,
                                "raan": 0.0,
                                "aop": 98.0,
                                "ta": 0.0
                            }
                        }
                }
    
    @abstractmethod
    def to_orbital_parameter_list(self) -> List[dict]:
        """ Converts constellation specifications to orbital parameters."""

    @abstractmethod
    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
        
    def propagate(self, propagation_period : float = None, out_dir : str = None, debug : bool = False) -> str:
        """ Propagates the constellation and saves results to specified output directory. """

        # define output directory
        out_dir = self.default_out_dir() if out_dir is None else out_dir
        
        # create output directory
        os.makedirs(out_dir, exist_ok=True)

        # define data filename
        data_filename = os.path.join(out_dir, 'MissionSpecs.json')
        
        # convert to orbital parameters
        orbital_params : List[dict] = self.to_orbital_parameter_list()

        # DEBUG print results
        if debug:
            print("="*80 + "\n")
            print(f"Walker Delta Constellation - i:t/p/f = {round(self.inc,2)}°:{self.num_sats}/{self.num_planes}/{self.phasing_param}")
            print("-"*82)
            prev_param = None
            for idx, params in enumerate(orbital_params):
                if prev_param and prev_param['raan'] != params['raan']:
                    print("  " + "-"*80)
                print(f"  Sat-{idx+1}:\tinc={params['inc']}°\traan={round(params['raan'],3)}°\taop={round(params['aop'],3)}°\tta={round(params['ta'],3)}°")
                prev_param = params
            print("="*80 + "\n")            

        # create satellite specification
        spacecraft = []
        for idx, params in enumerate(orbital_params):
            sat_spec = deepcopy(WalkerConstellation.SPACECRAFT_TEMPLATE)
            sat_spec['@id'] = f'sat_{idx+1}'
            sat_spec['name'] = f'sat_{idx+1}'
            sat_spec['orbitState']['state'] = params
            spacecraft.append(sat_spec)

        # calculate propagation period
        propagation_period = self.T / 2 / 3600.0 / 24.0 \
            if propagation_period is None else propagation_period

        # create mission specification
        mission_spec : Dict[str, dict] = deepcopy(WalkerConstellation.MISSION_TEMPLATE)
        mission_spec['duration'] = propagation_period 
        mission_spec['spacecraft'] = spacecraft
        mission_spec['settings']['outDir'] = out_dir

        if os.path.exists(data_filename):
            # load existing mission specifications
            existing_mission_spec : dict = json.load(open(data_filename,'r'))
            existing_mission_spec['propagator'].pop('stepSize', None)

            # compare with current specifications
            if existing_mission_spec == mission_spec: 
                print(f"Propagation already exists for `walker_delta_{self.num_sats}sat_{self.num_planes}planes_{self.phasing_param}phasing`.\n")
                return out_dir
            else:
                for key in mission_spec:
                    if existing_mission_spec[key] != mission_spec[key]:
                        print(f"Difference found in key: `{key}`")
                print(f"Existing propagation specifications differ from current specifications for `walker_delta_{self.num_sats}sat_{self.num_planes}planes`. Re-propagating...\n\n")
        else:
            print("Propagating orbits...")
        
        # create mission for propagation
        mission : Mission = Mission.from_json(mission_spec)  

        # propagate data and save to orbit data directory
        mission.execute(coverage_propagation=False, data_metrics_calculation=False,eclipse_finder=False)                
        print("Propagation done!")

        # delete state propagation from printed data for space savings
        for dir_name in os.listdir(out_dir):
            dir_path = os.path.join(out_dir,dir_name)
            if 'sat' in dir_name.lower() and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)            

        # remove step size from propagator for comparison purposes
        mission_spec['propagator'].pop('stepSize', None)

        # save specifications of propagation in the orbit data directory
        with open(data_filename, 'w') as mission_specs_file:
            mission_specs_file.write(json.dumps(mission_spec, indent=4))
        print(f"Saved mission specifications to \n   `{out_dir}`")

        # return output directory
        return out_dir   

class WalkerConstellation(Constellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes an abstract Walker constellation."""
        # validate inputs
        assert isinstance(alt, float) and alt >= 0.0, "Altitude must be a non-negative float"
        assert isinstance(inc, float), "Inclination must be a float"
        assert isinstance(num_sats, int) and num_sats > 0, "Number of satellites must be a positive integer"
        assert isinstance(num_planes, int) and num_planes > 0, "Number of planes must be a positive integer"
        assert isinstance(phasing_param, int) and 0 <= phasing_param, "Phasing parameter must be a positive integer"

        # calculate orbital period
        self.T = 2 * np.pi * np.sqrt( (Constellation.R + alt)**3 / Constellation.GM )

        # Store input parameters
        self.alt = alt
        self.inc = inc
        self.num_sats = num_sats
        self.num_planes = num_planes
        self.phasing_param = phasing_param

    def to_orbital_parameter_list(self) -> List[dict]:
        """
        Converts constellation specifications to a list of orbital parameters.

        - `i` is the inclination
        - `t` is the total number of satellites;
        - `p` is the number of equally spaced planes; 
        - `f` is the relative spacing between satellites in adjacent planes. 
                The change in true anomaly (in degrees) for equivalent satellites 
                in neighbouring planes is equal to f * 360° / t.
        """

        # initialize list of orbital parameters
        orbital_params = []

        # calculate number of satellites per plane 
        min_sats_per_plane = self.num_sats // self.num_planes
        remaining_sats_to_assign = self.num_sats % self.num_planes

        # calculate RAAN spacing and initial phasing
        raan_spacing = self.calc_raan_spacing()
        aop_phasing = self.phasing_param * (360.0 / self.num_sats)

        # generate orbital parameters for every satellite in every plane
        for plane_idx in range(self.num_planes):
            # calculate plane RAAN and initial aop
            raan = plane_idx * raan_spacing
            aop = plane_idx * aop_phasing 

            # calculate sats per plane (distribute remaining sats)
            sats_per_plane = min_sats_per_plane 
            sats_per_plane += 1 if plane_idx < remaining_sats_to_assign else 0

            # calculate intra-plane true anomaly spacing
            ta_spacing = 360.0 / sats_per_plane

            # generate orbital parameters for each satellite in plane
            for sat_idx in range(sats_per_plane):
                ta = sat_idx * ta_spacing

                params = {
                    "@type": "KEPLERIAN_EARTH_CENTERED_INERTIAL",
                    "sma": WalkerConstellation.R + self.alt,
                    "ecc": 0.001,
                    "inc": self.inc,
                    "raan": raan,
                    "aop": aop,
                    "ta": ta
                }
                orbital_params.append(params)

        return orbital_params
    
    @abstractmethod
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
     
class WalkerDeltaConstellation(WalkerConstellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes a Walker Delta constellation."""
        super().__init__(alt, inc, num_sats, num_planes, phasing_param)

    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
        return f'./orbits/walker-delta_{self.inc}inc_{self.num_sats}sat_{self.num_planes}pl_{self.phasing_param}f'
       
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
        return 360.0 / self.num_planes
    
class WalkerStarConstellation(WalkerConstellation):
    def __init__(self,
                 alt : float,
                 inc : float,
                 num_sats : int,
                 num_planes : int,
                 phasing_param : int
                ):
        """ Describes a Walker Star constellation."""
        super().__init__(alt, inc, num_sats, num_planes, phasing_param)
    
    def default_out_dir(self) -> str:
        """ Returns the default output directory for the constellation propagation. """
        return f'./orbits/walker-star_{self.inc}inc_{self.num_sats}sat_{self.num_planes}pl_{self.phasing_param}f'
       
    def calc_raan_spacing(self) -> float:
        """ Calculates the RAAN spacing between orbital planes."""
        return 180.0 / self.num_planes

if __name__ == "__main__":

    # terminal welcome message
    print_welcome(f'Walker Delta Constellation Example')
    
    # example usage
    walker = WalkerDeltaConstellation(
        alt = 550.0,
        inc = 98.0,
        num_sats = 8,
        num_planes = 3,
        phasing_param = 1
    )
    walker_data = walker.propagate(debug=True)