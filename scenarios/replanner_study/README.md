# Parameterized Study
Study aimed to understand the performance of preplanning strategies in an Earth-observing satellite mission for inland bodies of water.

## Results Analysis
### Observations
- Larger constellations out-perform smaller ones as expected
- More accesses opportunities and more points considered lead to more observations as expected
- Smaller reactive constellations with FIFO out-perform or have comparable performance to larger nadir-pointing constellations

### Ground Points Observed 
- More accesses leads to more GPs being observed, as expected
- Little data on large FIFO constellations
- Differences between preplanners is more noticible as the number of accessible ground-points increases.
- Most constellations seem to be able to see most of the ground-points accessible to them.
- Underperforming constellations consist of small (8 sats), nadir-pointing constellations. The constellations of the same size but with FIFO observe around 80-100% of all ground points.
- Ground-Points considered does not have a clear trend on performance of the constellations. This might be explained by the fact that access opportunities are very common and other satellites will plan to observe points that others are not. 

### Events Detected
- Event detection goes down as number of ground points are increased in the hydrolakes scenario but is pretty constant throughout the other grid distributions. 
- Large constellations tend to lead to larger number of events detected, as expected
- 

### Events Observed

### Events Re-Observed

### Events Co-Observed

### Event Observations

### Event Re-Observations

### Event Co-Observations

## General Trends
- Not enough runs were made with large constellations with the FIFO replanning strategy. Must've been the runs that did not finish running in time.
- 

## Candidate Scenarios for CBBA
> GOAL: Large number of events detected but not reobserved
### Candidate 1:
- Small constellations of 8 sats
- Large number of availbale and accessible ground-points
### Candidate 2:
- 24 nadir-pointing sat constellation on uniform grid

## Notes
- Something is wrong with the Co-observation results counter as histogram is always empty. Partial and full co-observations are properlly counted but total co-observations are not. Perhaps this happens when one of the two is a NAN and gets added to the real number? Needs fixing.
- Observation classifier uses observations to extract requested measureents. Should be from the events themselves. Make sure this has the same result.
- Ground-Point Access metrics for nadir-pointing sats over represent the accessibility if ground points for those constellations as the coverage calcs consider maneuverability of satellites, regardless of it's used or not. 
- Add metric of Events Accessible: an accessible event is one that has at least one access opportunity by a satellite with at least one of the instruments required to observe said event