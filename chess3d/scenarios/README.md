# 3DCHESS Scenarios 

## Creating a Scenario
1 - Create a directory within the `chess3d/scenarios` directory.

2 - In the newly created directory, create `MissionSpecs.json` file.

3 - If needed, create a new `resources` directory and add any additional scenario-specific data to said directory.

## MissionSpecs.json
This file contains all specifications outlining the scenario to be run. It must contain the following information:

### Epoch Description
This section outlines the start date of the simulation and scenario duration **in days**. The desired propagator for spacecraft orbit propagation can also be defined.

> Example:
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

### Spacecraft Description
Outlines the list of satellites present in the simulation. Describes the satellites components, orbit, instruments, planner settings, and science module capabilities. 

Satellites are not required for the simulation to be run.

> Example:
```
spacecraft": [
        {
            "@id": "imaging_sat_0_0",
            "name": "img_0",
            "spacecraftBus": {
                "name": "BlueCanyon",
                "mass": 20,
                "volume": 0.5,
                "orientation": {
                    "referenceFrame": "NADIR_POINTING",
                    "convention": "REF_FRAME_ALIGNED"
                },
                "components": {
                    "cmdh": {
                        "power": 2,
                        "memorySize": 100
                    },
                    "comms": {
                        "transmitter": {
                            "power": 3,
                            "maxDataRate": 1,
                            "bufferSize": 10,
                    ```    "receiver": {
                            "power": 3,
                            "maxDataRate": 1,
                            "bufferSize": 10
                        }
                    },
                    "eps": {
                        "powerGenerator": {
                            "@type": "Solar Panel",
                            "maxPowerGeneration": 10
                        },
                        "powerStorage": {
                            "@type": "Battery",
                            "maxPowerGeneration": 10,
                            "energyStorageCapacity": 0.01,
                            "depthOfDischarge": 0.99,
                            "initialCharge": 1
                        }
                    }
                }
            },
            "instrument": {
                "name": "visible",
                "mass": 10,
                "volume": 12.45,
                "dataRate": 40,
                "bitsPerPixel": 8,
                "power": 12,
                "snr": 33,
                "spatial_res": 50,
                "spectral_res": 7e-09,
                "orientation": {
                    "referenceFrame": "SC_BODY_FIXED",
                    "convention": "REF_FRAME_ALIGNED"
                },
                "fieldOfViewGeometry": {
                    "shape": "RECTANGULAR",
                    "angleHeight": 60,
                    "angleWidth": 60
                },
                "@id": "bs1",
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
                "horizon" : 3600,
                "preplanner" : "FIFO",
                "replanner" : "ACBBA",
                "dt_convergence" : 0.0,
                "utility" : "LINEAR"
            },
            "science" : "True",
            "notifier": "False",
            "missionProfile": "3D-CHESS"
        }
    ]
```

### UAVs 
Description pending

UAVs are not required for the simulation to be run.

### Ground Stations
Description pending. 

Ground Stations are not required for the simulation to be run.

### Grid
Description pending

> Example:
```
"grid": [
        {
            "@type": "customGrid",
            "covGridFilePath": "./scenarios/scenario2/resources/lake_event_points.csv"
        }
    ]
```

### Additional Scenario Settings
Description pending

> Example:
```
"scenario": {   
                    "@type": "PREDEF", 
                    "initialRequestsPath" : "./scenarios/scenario2_single_sat/resources/initial_requests_merged.csv",
                    "eventsPath" : "./scenarios/scenario2_single_sat/resources/all_events_formatted.csv", 
                    "duration": 30.0, 
                    "connectivity" : "FULL", 
                    "utility" : "LINEAR"
                },
```

### Additional Coverage Propagator Settings
Description pending

> Example:
```
"settings": {
        "coverageType": "GRID COVERAGE"
    }
```