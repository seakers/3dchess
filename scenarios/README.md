# 3DCHESS Scenarios 

This directory is meant to house any simulation scenario directories to be run. **Do not create new scenarios or directories within the `chess3d` directory.**

## Running a Scenario
The `chess3d` library provides a `Mission` class object that builds a scenario from a dictionary with mission specifications and runs it locally in the machine executing the script that calls it.

> Example simulation script:
```
if __name__ == "__main__":
    # define mision specicifations
    mission_specs : dict = {...}

    # initialize mission
    mission : Mission = Mission.from_dict(mission_specs)

    # execute mission
    mission.execute()
```


### Mission Specifications Format
This dictionarydescribes the specifications of the scenario to be run. It must contain the following information:

### 1. Epoch Description
This section outlines the start date of the simulation and scenario duration **in days**. The desired propagator for spacecraft orbit propagation can also be defined.

> **Example**:
```
"epoch": {
        "@type": "GREGORIAN_UT1",
        "year": 2020,
        "month": 1,
        "day": 1,
        "hour": 0,
        "minute": 0,
        "second": 0
    },
    "duration": 1,
    "propagator": {
        "@type": "J2 ANALYTICAL PROPAGATOR",
        "stepSize": 10
    },
```

### 2. Spacecraft Description
Outlines the list of satellites present in the simulation. Describes the satellites components, orbit, instruments, planner settings, and science module capabilities. 

Each satellite must define the following contents:
1. Satellite name and id
```
"@id": "SAT_ID,
"name": "SAT_NAME",
```
2. Instrument payload. Can be a list of dictionaries or a single dictionary
```
"instrument": {
    "name": "thermal",
    "mass": 10,
    "volume": 12.45,
    "dataRate": 40,
    "bitsPerPixel": 8,
    "power": 12,
    "snr": 33,
    "spatial_res": 50,
    "spectral_res": 7e-09,
    "orientation": {
        "referenceFrame": "NADIR_POINTING",
        "convention": "REF_FRAME_ALIGNED"
    },
    "fieldOfViewGeometry": {
        "shape": "RECTANGULAR",
        "angleHeight": 5,
        "angleWidth": 10
    },
    "maneuver" : {
        "maneuverType":"SINGLE_ROLL_ONLY",
        "A_rollMin": -50,
        "A_rollMax": 50
    },
    "@id": "therm1",
    "@type": "Basic Sensor"
}
```
3. Orbit state
```
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
        "sma": 7078,
        "ecc": 0.01,
        "inc": 67,
        "raan": 0.0,
        "aop": 0.0,
        "ta": 0.0
    }
}
```

Satellites may also include any of the following ptional parameters:
1. Planning Module Specifications. Defines the preplanning and (or) replanning strategies being used by the satellite. Reward grids can also be defined. If a preplanner or replanner are defined, no planner will be implemented. If no reward grid is explicitly defined, a fixed uniform reward grid will be generated.

```
"planner" : {
    "preplanner" : {
        "@type" : "naive"
    },
    "replanner" : {
        "@type" : "acbba",
        "bundle size" : 3
    },
    "rewardGrid":{
        "reward_function" : "event",
        "initial_reward" : 1.0,
        "min_reward" : 1.0,
        "unobserved_reward_rate" : 2.0,
        "max_unobserved_reward" : 10.0,
        "event_reward" : 10.0
    }
}
```
2. Science Module Specifications:
```
"science" : {
    "@type": "lookup", 
    "eventsPath" : "./scenarios/algal_blooms_study/resources/random_events.csv"
}
```

Satellites are not required for the simulation to be run.

> **Example**:
```
spacecraft": [
        {
            "@id": "thermal_sat_0_0",
            "name": "thermal_0",
            "spacecraftBus": {
                "name": "BlueCanyon",
                "mass": 20,
                "volume": 0.5,
                "orientation": {
                    "referenceFrame": "NADIR_POINTING",
                    "convention": "REF_FRAME_ALIGNED"
                },
                "components": {
                    "adcs" : {
                        "maxTorque" : 1000,
                        "maxRate" : 1
                    }
                }
            },
            "instrument": {
                "name": "thermal",
                "mass": 10,
                "volume": 12.45,
                "dataRate": 40,
                "bitsPerPixel": 8,
                "power": 12,
                "snr": 33,
                "spatial_res": 50,
                "spectral_res": 7e-09,
                "orientation": {
                    "referenceFrame": "NADIR_POINTING",
                    "convention": "REF_FRAME_ALIGNED"
                },
                "fieldOfViewGeometry": {
                    "shape": "RECTANGULAR",
                    "angleHeight": 5,
                    "angleWidth": 10
                },
                "maneuver" : {
                    "maneuverType":"SINGLE_ROLL_ONLY",
                    "A_rollMin": -50,
                    "A_rollMax": 50
                },
                "@id": "therm1",
                "@type": "Basic Sensor"
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
                    "sma": 7078,
                    "ecc": 0.01,
                    "inc": 67,
                    "raan": 0.0,
                    "aop": 0.0,
                    "ta": 0.0
                }
            },
            "planner" : {
                "preplanner" : {
                    "@type" : "naive"
                },
                "replanner" : {
                    "@type" : "broadcaster"
                },
                "rewardGrid":{
                    "reward_function" : "event",
                    "initial_reward" : 1.0,
                    "min_reward" : 1.0,
                    "unobserved_reward_rate" : 2.0,
                    "max_unobserved_reward" : 10.0,
                    "event_reward" : 10.0
                }
            },
            "science" : {
                "@type": "lookup", 
                "eventsPath" : "./scenarios/algal_blooms_study/resources/random_events.csv"
            }
        }
    ]
```

### 3. UAVs 
Description pending

UAVs are not required for the simulation to be run.

### 4. Ground Stations
Description pending. 

Ground Stations are not required for the simulation to be run.


### 5. Scenario Configuration
Outlines specific parameters for the simulation. Particularly, it defined the network conectivity between satellites. By default this is set to "FULL" in which agents are assumed to have constant communications with eachother. setting connectivity to "LOS", the simulation will only allow for inter-agent communications in times in which agents are in line-of-sight of one and other. 

This section also defines the events present in the simulation. These can be randomly generated at the start of the simulation, or predefined from an external `csv` file and imported in the simulation. 

The Scenario Path defines the location of the scenario directory. The name parameter determines the name of the simulation about to be run. This will be reflected in the name of the directory containing the results of said simulation.

```
"scenario": {   
    "connectivity" : "FULL", 
    "utility" : "LINEAR",
    "events" : {
        "@type": "random", 
        "numberOfEvents" : 1000,
        "duration" : 3600,
        "minSeverity" : 0.0,
        "maxSeverity" : 100,
        "measurements" : ["sar", "visual", "thermal"]
    },
    "clock" : {
        "@type" : "EVENT"
    },
    "scenarioPath" : "./scenarios/algal_blooms_study/",
    "name" : "toy"
}
```

### 6. Additional Scenario Settings
Miscelanous settings primeraly related to the `orbitpy` propagation library. Defines the type of coverage being calculated as well as the directory that will contain the resulting orbit propagation and access times for this simulation.

```
"settings": {
    "coverageType": "GRID COVERAGE",
    "outDir" : "./scenarios/algal_blooms_study/orbit_data"
}
```

### 7. Grid
Defines the grid of Ground Points being used to calculate coverage. These can be generated at the start of the simulation via specified parameters, or predefined from an external `csv` file and imported in the simulation. 

> Example:
```
"grid": [
    {
        "@type": "customGrid",
        "covGridFilePath": "./scenarios/scenario2/resources/lake_event_points.csv"
    }
]
```
