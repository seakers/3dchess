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


### Test Matrix
| Parameter | Values | Units |
|-----------|--------|-------|
| Mission Duration | $1$ | $\text{[days]}$ |
| Connectivity | $\text{LOS}$, $\text{Full}$ | - |



### Mission Definition
#### Mission 1 - Reactive Event Scheduling
##### Default Mission Objectives
*None*
##### Event-Driven Mission Objectives
1. Response Time
    - Pass
2. Revisit Time 
3. Co-Observation Time
4. Observation Number


