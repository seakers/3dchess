# ACBBA RUNTIME OPTIMIZATION
These scenarios were created with the goal of improving the runtime of the ACBBA algorithm. Prior to this work, simulations of 3D-CHESS' Scenario 2 would take 1.5 hrs for a bundle size of 1 and up to 5 hours for a bundle size of 3.

## Runtime Analysis

### Bundle-size: 1
|  Routine              | n_calls | t_avg | t_med | t_std | t_total |
|-----------------------|---------|-------|-------|-------|---------|
| `generate_plan()`     | 86      | 3.312 | 17.505| 1.333 | 284.832 |
| `planning_phase()`    | 86      | 3.214 | 17.211| 1.289 | 276.404 | 
| `calc_path_bid()`     | 3288    | 0.033 | 0.098 | 0.02  | 108.504 |
| `needs_planning()`    | 1569    | 0.044 | 0.163 | 0.014 | 69.036  |
| `calc_imaging_time()` | 3288    | 0.016 | 0.057 | 0.01  | 52.608  |
| `consensus_phase()`   | 1569    | 0.007 | 0.156 | 0.0   | 10.983  |
| `plan_from_path()`    | 86      | 0.096 | 0.301 | 0.044 | 8.256   |
| `schdule_measurements`| 86      | 0.001 | 0.007 | 0.0   | 0.086   |
| `schedule_broadcasts` | 86      | 0.048 | 0.18  | 0.033 | 4.128   |
| `schedule_maneuvers`  | 86      | 0.014 | 0.03  | 0.01  | 1.204   |

Most of the runtime is consumed during the planning phase of the algorithm. This is when a bundle is constucted and communicated to the rest of the constellation. 

### Recommendations for improvement
- `generate_plan()` takes up the most runtime, therefore limiting the number of times this method is called would have the most impact in overall runtime. For a simulation that generated 32 task-requests, replanning was called 86 times. `needs_replanning()` needs to be revised to ensure replanning is done in the appropriate times (see Luke Johnson's ACBBA paper for guidance).

- `generate_plan()`'s runtime is split between `planning_phase()` and `plan_from_path()`, improving their respective runtimes will have an overal effect on runtime.

- `plan_from_path()` is a much simpler method than `planning_phase()` and offers an easy opportunity to improve runtime. 

- `planning_phase()`'s runtime indicate that it is the main source of runtime for any simulation. This is due to the recursive nature of the bundle-building approach implemented in this method. `calc_path_bid()` (which is called by `planning_phase()`) seems to be called the most out of any of the analized methods. Although its runtime per iteration is not too severe (0.033 [sec/iter]), it is called twice as many times as its parent method. The following actions are suggested:
    - Revisions to the end-condition for the `while`-loop in use should be reconsidered.
    - Selection of max bid should levarage built-in python methods to improve runtime.
    - Reduction of search space for tasks should be implemented. Suggested approaches include the implementation of a planning horizon (might require changes to `needs_replanning()` to accomodate for actions that are available and have just entered the planning horizon).
    - Implement a bundle-building approachthat does not require full factorial search of the search space.


## Scenarios

1) Full Scenario 2 w/N=1 

### Modifications Implemented
#### `schedule_broadcasts()`: 
- Removed the need to create new instances of `Bid`s and `MeasurementRequest`s when only minimal data was needed from them. Used their dictionary form instead. 
- Removed redundant loops and collapsed them into a single loop using python's built-in filtering functionality.
- Run-time improvement: ~0.01[sec/iter]

#### `needs_replanning()`:
- Attempted to implement periodic replanning but this increased runtime. This idea was abandoned and it was chosen to maintian event-oriented structure.
- Allowed for bids to accumulate in an internal buffer and only trigger replanning once their number exceed a given threshold (set to a default of 1). This resembles Luke Johnson's implementation 
