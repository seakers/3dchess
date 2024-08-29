# 3DCHESS: Decentralized, Distributed, Dynamic, and Context-aware Heterogeneous Sensor Systems

Repository containing tools related to NASA's 3DCHESS AIST project. 

This application simulates a distributed sensor web Earth-Observation system described in the
3D-CHESS project which aims to demonstrate a new Earth
observing strategy based on a context-aware Earth observing
sensor web. This sensor web consists of a set of nodes with a
knowledge base, heterogeneous sensors, edge computing,
and autonomous decision-making capabilities. Context
awareness is defined as the ability for the nodes to gather,
exchange, and leverage contextual information (e.g., state of
the Earth system, state and capabilities of itself and of other
nodes in the network, and how those states relate to the dy-
namic mission objectives) to improve decision making and
planning. The current goal of the project is to demonstrate
proof of concept by comparing the performance of a 3D-
CHESS sensor web with that of status quo architectures in
the context of a multi-sensor inland hydrologic and ecologic
monitoring system.

## Directory structure
```
├───.temp (To be determined, temporary folder with old files)
├───docs (sphinx and other documentation)
├───scenarios (folder with the input and results files for simulations)
├───tests (unit tests)
└───chess3d (folder with main source code)
    ├───nodes (dmas network nodes)
    └───agents (implementations for all agents)
        ├───engineering (engineering module files)
        ├───science (science module files)
        └───planning (planning module files)
            ├───.backup (To be determined, temporary folder with old files)
            └───planners (planner implementation files)
                └───conseunsus (consensus-based planners)
```


## Installation
1. Install the [`dmas`](https://github.com/seakers/DMASpy) library and all of its dependencies is required for running these scenarios.

2. Run `make` from the root repo directory.

## Running a Simulation

Define 

<!-- ## Running DMAS - 3DCHESS

Open a terminal in this directory and run `main.py` by entering the following command:

    python main.py <scenario name>

To create a scenario, see `scenarios/README.md`. -->

## Acknowledgments

This work has been funded by grants from the National Aeronautics and Space Administration (NASA) Earth Science Technology Office (ESTO) through the Advanced Information Systems Technology (AIST) Program.

## Contact 
**Principal Investigator:** 
- Daniel Selva Valero - <dselva@tamu.edu>

**Lead Developers:** 
- Alan Aguilar Jaramillo - <aguilaraj15@tamu.edu>
- Ben Gorr - <bgorr@tamu.edu>
