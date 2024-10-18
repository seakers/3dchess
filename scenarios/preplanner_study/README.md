# Parameterized Study
Study aimed to understand the performance of preplanning strategies in an Earth-observing satellite mission for inland bodies of water.

## Coverage Grid
All simulations use the same grid of lakes and from the  HydroLAKES dataset.

## Parameter space
| Parameter                  | Values                      |
|----------------------------|-----------------------------|
| Constellateion (n_p, n_s)  | (1,8), (2,4), (3,4), (8,3)  |
| Field of Regard (deg)      | 30, 60                      |
| Field of View (deg)        | 1, 5, 10                    |
| Maximum Slew Rate (deg/s)  | 1, 10                       |
| Number of Events per Day   | 10, 100, 1000, 10000        |
| Event Duration (hrs)       | 0.25, 1, 3, 6               |

## Planning Stretegies
- Dynamic Programmer Periodic Planner
- Dynamic Programmer Periodic Planner + CBBA replanner
- FIFO preplanner
- FIFO preplanner + CBBA replanner
- Greedy Planner
- Forward-search Planner
- MILP Planner