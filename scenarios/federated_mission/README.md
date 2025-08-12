# Federated Mission Scenario

## Mission 1 - Algal blooms

### Agents
- 1 Hyperspectral VNIR 
- 2 TIR
- 5 Altimeter

### Objectives

#### $O_{1,1}$ - Measure Chlorophyll-A ($w=1$)
- $MR_{1,1,1}$ - Horizontal spatial resolution: $Th = [10, 30, 100]m$, $Sc = [1.0, 0.7, 0.1]$
- $MR_{1,1,2}$ - Spectral resolution: $Th = [Hyperspectral, Multispectral]$, $Sc = [1, 0.5]$

#### $O_{1,2}$ - Measure Water temperature ($w=1$)
- $MR_{1,2,1}$ - Horizontal spatial resolution: $Th = [30, 100] m$, $Sc = [1.0, 0.3]$

#### $O_{1,3}$ - Measure Water level ($w=1$)
- $MR_{1,3,1}$ - Horizontal spatial resolution: $Th = [30, 100] m$, $Sc = [1.0, 0.5]$
- $MR_{1,3,2}$ - Accuracy: $Th = [10, 50, 100] cm$, $Sc = [1.0, 0.5, 0.1]$

#### $O_{1,4}$ - Observe events “Algal Blooms” ($w=10$)
Main parameters and requirements:
- **Chl-A**, $R = \{MR | MR \in O_{1,1}\} $
- **Water Temperature**, $R = \{MR | MR \in O_{1,2}\} $
- **Water Level**, $R = \{MR | MR \in O_{1,3}\} $

### Detection Algorithm
See slides (Chl-A from VNIR radiances using formula, then compare to historical values for that location)

### Characterization Algorithm
Severity proportional to lake area (as in paper) 

### Re-Observation Strategy
From Ben’s paper, rewards for subsequent observations or something simple like $U(n)$ first increases to guarantee some reobs but then decreases exponentially beyond a certain $n_{obs}$ (e.g., 3)


## Mission 2 - Floods and sediment transport TSS
#### Agents 3 
- 3 Multispectral VNIR
- 4 Altimeter

#### $O_{2,1}$: Measure Water level ($w=1$)
- $MR_{2,1,1}$ - Horizontal spatial resolution: $Th = [30, 100]m$, $Sc = [1.0, 0.5]$
- $MR_{2,1,2}$ - Accuracy: $Th = [10, 50, 100] cm$, $Sc = [1.0, 0.5, 0.1]$

#### $O_{2,2}$: Measure turbidity ($w=1$)
- $MR_{2,2,1}$ - Horizontal spatial resolution:$ Th = [30, 100] m$, $Sc = [1.0, 0.3]$
- $MR_{2,2,2}$ - Spectral resolution: $Th = [Hyperspectral, Multispectral] m$, $Sc = [1, 0.5]$

#### $O_{2,3}$: Measure TSS ($w=1$)
- $MR_{2,3,1}$ - Horizontal spatial resolution: $Th = [30, 100] m$, $Sc = [1.0, 0.3]$
- $MR_{2,3,2}$ - Spectral resolution: $Th = [Hyperspectral, Multispectral] m$, $Sc = [1, 0.5]$

#### $O_{2,4}$: Observe events “High flow event” ($w=10$)
Main parameters and requirements:
- **Water Level**, $R = \{MR | MR \in O_{2,1}\} $
- **Turbidity**, $R = \{MR | MR \in O_{2,2}\} $
- **TSS**, $R = \{MR | MR \in O_{2,3}\} $

### Detection Algorithm
See slides (compare to historical values)

### Characterization Algorithm
Severity proportional to lake area (as in paper) 



### Re-Observation Strategy
From Ben’s paper, rewards for subsequent observations or something simple like $U(n)$ first increases to guarantee some reobs but then decreases exponentially beyond a certain $n_{obs}$ (e.g., 3)
