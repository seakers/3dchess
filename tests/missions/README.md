# Mission Tests
Unit tests for mission definitions contained in `chess3d/missions` directory.

## Multi-mission scenario
Proposed test case for full federated mission capability testing. 

### Algal blooms mission 
| Instrument | Number of Agents | 
|-|-|
| VNIR hyp | 1 |
| TIR | 2 |
| Altimeter | 5 |

| Objective | Parameter | Weight | Requirement | Threshold | $u(x)$  |
| ----------|-----------|--------|-------------|-----------|-------- | 
| $O_1$ | Measure Chlorophyll-A  | $w=1$ | Horizontal spatial resolution | [10, 30, 100] m | [1.0, 0.7, 0.1] |
| " | " | " | Spectral resolution | [Hyperspectral, Multispectral] | [1, 0.5] |
| $O_2$ | Measure Water temperature | $w=1$ | Horizontal spatial resolution | [30, 100] m | [1.0, 0.3] |
| $O_3$ | Measure Water level | $w=1$ | Horizontal spatial resolution | [30, 100] m | [1.0, 0.5] |
| " | " | " | Accuracy | [10, 50, 100] cm | [1.0, 0.5, 0.1] |
| $O_4$ | Observe events “Algal Blooms” | $w=10$ | Chl-A Horizontal spatial resolution | [10, 30, 100] m | [1.0, 0.7, 0.1] |
| " | " | " | Chl-A Horizontal spatial resolution | [30, 100] m | [1.0, 0.3] |

<!-- #### Objectives

O4: Observe events “Algal Blooms” (w=10)
Main parameter: Chl-A, MR = O1

DA: See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)
CA: Severity proportional to lake area (as in paper) 
CO: Secondary params = Water temperature and water level, MR as in O2 and O3
RO: From Ben’s paper, rewards for subsequent observations or something simple like U(n) first increases to guarantee some reobs but then decreases exponentially beyond a certain #obs (e.g., 3) -->

### Floods and sediment transport TSS mission: 
**Agents** 3 (VNIR multi) and 4 (Altimeter)

O1: Measure Water level (w=1)
Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.5]
Accuracy: Thresholds = [10, 50, 100] cm, u = [1.0, 0.5, 0.1]
O2: Measure turbidity (w=1)
Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.3]
Spectral resolution: Thresholds = [Hyperspectral, Multispectral] m, u = [1, 0.5]
O3: Measure TSS (w=1)
Horizontal spatial resolution: Thresholds = [30, 100] m, u = [1.0, 0.3]
Spectral resolution: Thresholds = [Hyperspectral, Multispectral] m, u = [1, 0.5]
O4: Observe events “High flow event” (w=10)
Main parameter: Water level, MR = O1
DA: See slides (compare to historical values)
CA: Severity proportional to lake area (as in paper) 
CO: Secondary params = Water temperature and water level, MR as in O2 and O3
