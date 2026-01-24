# CBBA Internal Validation Stress Test
*GOAL: showcase the reactivity capabilities of the sequence-constrained CBBA to incoming event requests and test edge cases*

## Experiment Formulation
### Research questions
1. How many requests can this CBBA handle?
    1. See runtime issues
    2. How can you guarantee or show convergence?
2. How efficiently does it assign tasks?
    1. How many messages are required to coordinate observations?
    2. How much agreement is actually achieved?

### Simulation Parameters
| Parameter | Values | Units |
|-----------|--------|-------|
| | |
| **Event Parameters** |  |  |
| Event Intensity | $\text{Uniform}(5.0,10.0)$ | - |
| Event Duration  | $\text{Uniform}(5,15)$ | [min] |
| Longitude Target Distributrion | $\text{Uniform}(-180, 180)$  | [deg]
| Target Location Condition | Inland Target | - |
| Decorrelation Time ($t_{corr}$) | $\text{Uniform}(1,10)$ | [min] |
| | |
| **Planner Parameters** | | |
| Replanning Threshold | 1 | [tasks] |
| Optimistic Bidding Threshold ($\nu$) | 1 | - |
| | |
| **Agent Capability Parameters** | | |
| Communications Range | $\text{LOS}$ | - |
| Maximum Slew Rate | $15$ | [deg/s] |
| Instrument | `IMG_A`, `IMG_B`, `IMG_C` | - |

### Test Matrices
#### Trial 1 - Stress Test in Fully Connected Network w/ Varying Latency
| Parameter | Values | Units |
|-----------|--------|-------|
| Number of Satellites | $12, 48, 96, 204$ | - |
| Constellation Connectivity Infrastructure | Alaska Satellite Facility, Full NEN , ISL | - | 
| Task arrival-rate ($\lambda$) | $10, 100, 500, 1000$ | [tasks / day] | 
| Latitude Target Distribution | $±25, ±60, ±90$ | [deg] |
| | |
**Total Cases:** 144

## Running Trials

```
python ./experiments/1_0_cbba_stress_test/study.py -n full_factorial_trials -o False -l LOWER -u UPPER
```

<!-- ### Mission Definition
#### Mission 1 - Reactive Event Scheduling
##### Default Mission Objectives
*None*
##### Event-Driven Mission Objectives
1. Response Time
    - Pass
2. Revisit Time 
3. Co-Observation Time
4. Observation Number -->


