# Mission Specification: Automating science operational decisions

Each agent has a **mission** $M=\{O_1, \text{...} ,O_m\}$ defined a s a set of objectives. 
An **objective** is defined as a tuple $O=⟨w,param,R⟩$ where $w$ is the relative weight of that objective to the agent’s mission; $param$ is the primary geophysical parameter/data product (e.g., chlorophyll-Aconcentration); and $R=\{MR_1,...,MR_n\}$ is a set of measurement requirements. 

## Mission Requirements for Attribute Representation

A (measurement) **requirement** is defined by a tuple $MR=⟨At,p(x)⟩$ where $At$ is an attribute of the data product (e.g., horizontal spatial resolution) and $p(x) : X \rightarrow [0,1]$ is a **preference function** that relates values of a given requirement's attribute to a fitness score between 1 and 0. The definition of $p(x)$ can vary depending on the nature of the attribute being characterized by the requirement. 
 <!-- For example, for spatial resolution, $Th=[10, 30, 100]$ and $Sc=[1.0, 0.75,0.25]$, one can fit a any smooth monotonically decreasing function, such as a sigmoid or generalized rational decay.  -->

<!-- ### Spatial and temporal coverage and sampling -->
<!-- Spatiotemporal coverage and sampling are by far the most common attributes on which requirements will often be specified.  -->

- **Spatial coverage** requirements define the region of interest for the observations, which can be specified as a list of *lat, lon* points or as a coverage grid defined over a region of interest. For regions larger than a point, spatial sampling can be specified with individual requirements for in horizontal and or vertical spatial resolution, as applicable. 
    - $p(x)$ in this case could be a binary preference function $p(x) \in \{0,1\}$ that returns $1$ if any of the desired targets $x_{target} \in X_{targets}$ are within the observed targets $x$ and $0$ otherwise.
    - A different preference function could also be selected such that it benefits the simultaneous observation of multiple desired targets by returning the percentage of desired targets currently being observed $p(x) = |x \cap X_{targets}|/|X_{targets}|$.

- **Temporal coverage** requirements define the extent of time during which observations are to be performed. For continuous monitoring objectives, this maybe a long time(e.g., a month), whereas for event-based objectives, it may be a much shorter duration depending on the event type (e.g., a few hours). During this time window, the temporal sampling with which observations are to be performed can be specified. 
    - For continuous monitoring objectives, one can simply define a **revisit time requirement** with the corresponding preference function. Note that the preference function specified for revisit time implicitly provides a way to value re-observations of the same point or region. The marginal value of the next observation is the slope of the preference function at the current revisit time. 
    - Alternatively, one can explicitly provide a preference function as a function of the number of observations of the same point. The latter can be more natural to define for event-based objectives.

- **Observation performance attributes** can also be used to specify requirements. Some of the most common ones may be spectral coverage and sampling, radiometric resolution, or signal-to-noise ratio. Higher-level attributes can also be used, such as day-night or all-weather capability. Continuous or discrete preference functions can be defined for these and other attributes using the method described above, even for more qualitative attributes as demonstrated in the VASSAR papers (Selva et al., 2014; Selva & Crawley, 2013). 
    - One approach could include the use of threshold and score values ($Th$ and $Sc$ respectively) to define a preference mapping function. Specifically, $Th$ could be an ordered list of threshold values $[x_1=x_{best},x_2,...,x_{worst}]$ and $Sc$ is an ordered list of scores $[u_1=u_{best},u_2,...,u_{worst}]$. 
        - For discrete attributes, $Th$ may contain all possible values and $Sc$ the score for each value. 
        - For continuous attributes, $p(x)$ is interpolated from $Th$, $Sc$. For example, a **spatial resolution** requirement could be chacarterized by a smooth monotonically decreasing function, such as a sigmoid or generalized rational decay, fitted to some threshold and score values (i.e., $Th=[10, 30, 100]$ and $Sc=[1.0, 0.75,0.25]$).

- **Capability attributes** characterize whether a satellite’s instruments or control systems can support the observation of a desired geophysical parameter. These capabilities can be represented through **measurement requirements**, using preference functions that quantify how well an agent’s capabilities align with those required by a mission objective. 

    - For example, the ability of an instrument $x$ to observe a given objective's primary parameter can be modeled as a simple membership check: whether $x$ belongs to a reference set of valid instruments $X_{ref}$. A binary preference function $p(x) = \mathbb{1}_{x \in X_{ref}}$ can then express the instrument’s fitness for the objective.


    - This functionality and its evaluation can be facilitated and expanded through knowledge graphs or other formal knowledge representations.


## Geophysical Events of Interest
An event of interest can be detected by an agent. An **event** $E$ is defined by a tuple $E =⟨eventType, loc, t_{detect}, d_{exp}, S⟩$. It is initially characterized by a type, location $loc$, and detection time $t_{detect}$ . On board data processing is used to determine an expected duration $d_{exp}$ and severity $S$. 

## Generalized Task Definition
When scheduling operations, satellites require a set of observation tasks that can be evaluated for performance, prioritized, and ultimately assigned. In a constellation, these tasks should also be shareable so that all satellites operate from a common pool of known tasks.

To enable sharing, tasks are defined in two tiers:

- **Generic Observation Task**: a high-level definition containing general information about the target and the observation objectives. This representation is shared across the constellation.

- **Specific Observation Task**: a satellite-specific version of a generic task, enriched with details relevant only to that satellite (e.g., required attitude, instrument used, access time window, measurement duration, etc.). This information is not shared and is used exclusively for that satellite’s scheduling process.

### Generic Observation Tasks
Generic observation tasks are high-level representations that define **what** should be observed, **where**, and **when**, along with a **priority** and the **mission or event context**.  

Each task can be represented as:  
$$\tau = \langle param, loc, time, r, context \rangle$$

Where:  
- $param$ - the geophysical parameter (or desired measurement type) to be observed  
- $loc$ - the geographic location of the observation  
- $time$ - the specific time or time interval when the observation is valid  
- $r$ - the observation priority  
- $context$ - a reference to the mission objective(s) and/or event information that motivated the task  

These tasks may originate from nominal mission objectives or from detected geophysical events, and their sharing or exclusivity depends on the type of task.

#### Default Mission Tasks  
<!--#### Default Mission Tasks
During nominal operations, satellites must schedule observations based on their default mission objectives. Any possible observation window is then represented as a default observation task $\tau_{default}=⟨param, loc, time, r, O_{ref}⟩$ which indicate the the relevant parameter to be obeserved at location $loc$ at a time or time interval $time$ with a priority $r$ to satisfy a given mission objective $O_{ref}$. These tasks are not shared between agents and are only considered by the agent generating them based on its own mission objectives.  -->

During nominal operations, satellites generate **default mission tasks** from their predefined mission objectives. Any possible observation target matching these objectives is expressed as:  

$$\tau_{default} = \langle param, loc, [0, \infty), r, O_{ref} \rangle$$

Where $O_{ref}$ is the mission objective driving the observation. These tasks are *not shared* between agents; they are considered only by the agent that generated them. Default mission tasks never expire, hense their availability interval is defined as $ time = [0,\infty)$

#### Event-Driven Tasks  
<!--#### Event-Driven Tasks
To respond to event $E$, one or more tasks $\tau_{event}=⟨param, loc, time, r, [O_{ref},E]⟩$ are generated by the agent that discovered the event which require measuring the relevant parameters at location $loc$ and time or time interval $time$, with a priority of $r$. This priority is tied to the severity of the event being observed. In addition, the agent can include the relevant objective $O_{ref}$ from its mission and the description of the detected event $E$ in the task, to include any relevant requirements. 

One task is created for each event's geophysical parameter (or desired instrument types, by using level 1 data products such as TIR radiances). The requirements for cross-registered co-observations of the event (of different parameters, or from different sensor types) are managed by creating the corresponding task requests with the same location and time intervals.  -->

When an event $E$ is detected, one or more **event-driven tasks** are generated:  

$$\tau_{event} = \langle param, loc, time, r, [O_{ref}, E] \rangle$$

The priority $r$ reflects the severity of the event, and the context includes both the relevant mission objective(s) $O_{ref}$ and the event description $E$.  

Because event-driven tasks can be shared between agents, there may be security or privacy concerns when including detailed context. In such cases, an agent may choose to omit the mission objective and event description, creating a **context-free** event-diven task:  

$$\tau_{event} = \langle param, loc, time, r, \emptyset \rangle$$

When a context-free task is received, agents must interpret it using their own mission objectives and knowledge to assign a relevant context before considering it for scheduling.

For events requiring **cross-registered co-observations** (e.g., multiple parameters or different sensor types), separate tasks are created with the same location and time interval but different $param$ values.  

### Specific Observation Tasks
When scheduling observation tasks, additional detailed information is often necessary to accurately predict the feasibility of a plan and to evaluate its overall quality. Additionaly, due to the characteristics of an instrument’s field of view, it is possible to observe multiple targets simultaneously, effectively satisfying multiple generic observation tasks in one observation. These combined observations can offer greater value during scheduling and thus should be represented as distinct specific tasks. 

Using a set of known Generic Observation Tasks, along with their capabilities and access times, agents generate Specific Observation Tasks enriched with this additional information. A Specific Observation Task can be defined as:

$$\sigma = \langle \Tau_{parent}, I, \Theta, TW, R_{duration} \rangle$$

Where:

- $\Tau_{parent}$ - Set of parent generic tasks  
- $I$ - Instrument assigned to perform the observation  
- $\Theta$ - Range of feasible attitude orientations for the observation  
- $TW$ - Accessibility time window during which the observation can be scheduled  
- $R_{duration}$ - Required observation duration



Initially, one specific task is generated for each known generic task. Subsequently, additional specific tasks can be created by clustering existing specific tasks to capture opportunities for simultaneous observations, as described in Wu et al. (2013).


<!-- ## Mission Objectives
### Default Mission Objectives

### Event-based Objectives
The mission specification described above is flexible enough to evaluate observations related to both regular objectives, related to the default mission of the agent, and event-based objectives, related to any task requests received to respond to events of interest. To see this, let us consider how the agent will value a specific task request during operations given its mission specification.  -->

## Evaluating Tasks using Mission Objectives
#### Mapping Objectives to Generic Tasks
Given a generic observation task $j$, the agent must identify or create a relevant event-based objective $k$. This is captured by the matrix $Rel_{jk} \in [0, 1]$ which maps tasks to objectives by assigning a fitness value where the higher the value, the more relevant an objective is to a given task.

If all agents know a priori about all possible task and event types and have objectives defined for each of them, the creation of this matrix is trivial and the corresponding objectives can be used as is. If agents have different objectives and the agent receiving a task request does not have an objective related to that event, it must create a new objective. The weight of the new objective can be determined based on the weight of the closest existing objective in terms of semantic similarity, using ametric supported by a large language model that leverages the KG. The rest of the new event-based objective information can be copied from the task request, if it is provided. Otherwise, it can also be inferred using a similar approach. For example, it is reasonable to assume that to respond to events of interest about rivers, one may need similar spatial resolution. 

#### Evaluing Task Performance
Once a generic task is mapped to an existing or new objective, its value can be determined by relating its performance to an agent's mission objectives. The task and the agent together determine the geometry of the observation, which can be used to compute the performance attributes $x_{ijl}$. The performance of an observation $P_{ijk}$ is then defined as the product of the utilities of all the performance attributes $p_{ikl}(x_{ijl})$ by agent $i$ for task $j$ for all for all mission requirements $l \in R(O_k)$ in a given mission objective $k$:

$$  P_{ijk} = \prod_{l \in R(O_k)}  p_{ikl}(x_{ijl}) $$

 <!-- multiplied by the relevance of a task to the given objective $Rel_{jk}$.  -->

#### Calculating Task Value 
Since some generic tasks have higher priority than others, the preference functions are objective-specific, and not all tasks are relevant to all objectives, the value $V_{ij}$ of agent $i$ performing generic task $j$ to the agent’s objective is computed as:

<!-- From this, we can compute the total scientific value of agent $i$ performing task $j$ as the weighted sum of all objective values: -->

$$
V_{ij} = r_{j} \cdot \sum_{k \in M} w_{ik} \cdot Rel_{jk} \cdot P_{ijk} 
$$

#### Calculating Specific Task Utility
Finally, for the purposes of task planning and scheduling, we may want to consider the cost of performing a task in addition to its scientific value. Therefore, specific observation tasks are evaluated by combining the aggregated value of their parent generic tasks with the cost of executing the specific task, forming a utility function.

In the context of satellite constellations, this cost typically corresponds to the energy expenditure required for the slewing maneuver needed to point the satellite’s instrument toward the task location. Since satellites can regenerate energy through solar panels, it is reasonable to assign this cost a small but non-zero weight—enough to encourage efficient use of resources without overly penalizing necessary maneuvers.

Accordingly, the utility $U_{in}$ of satellite $i$ performing specific task $n$ is defined as:
$$U_{in} = \sum_{j \in \Tau_n} V_{ij} - \alpha E_{in} $$

Where:
- $V_{ij}$ is the value of parent generic task $j$ to satellite $i$,
- $\Tau_{n}$ is the set of parent generic tasks of the specific task $n$,
- $E_{in}$ is the energy cost for satellite ss to perform task $n$,
- $\alpha$ is a small positive coefficient balancing the importance of energy cost.

<!-- Finally, for the purposes of task planning and scheduling, we may want to consider the cost of performing a task in addition to its scientific value. Therefore, **specific observation tasks** are evaluated instead of generic tasks by using the sum of the values of their parent tasks along with the cost of performing said specific task to define a utility funciton. 

For satellite constellations, this would typically represent the cost of the slewing maneuver needed to point at the location of the task. In that case, it is reasonable to use a very small (but non-zero) value since energy can be regenerated “for free” with solar panels, but we still want to incentivize efficiency in the plan. 

The utility of satellite $s$ for performing specific task $n$ is therefore defined as:

$$U_{in} = \sum_{j}^{\tau_j \in \Tau_n} V_{ij} - \alpha E_{in} $$ -->

## Definitions Summary
| Expression | Definition |
|------------|------------|
| $M=\{O_1, \text{...} ,O_m\}$ | Mission |
| $O=⟨w,param,R⟩$ | Objective |
| $w$ | Relative objective weight |
| $param$ | Primary geophisical parameter/data product |
| $R=\{MR_1,...,MR_n\}$ | Objective Requirements |
| $MR=⟨At,Th,Sc⟩$ | Measurement Requirement | 
| $At$ | Attribute of the data product (e.g., horizontal spatial resolution) |
| $x$ | Performance attribute |
| $p(x) : X \rightarrow [0,1]$ | Attribute preference function (maps performance attribute $x$ to requirement satisfaction value between $[0,1]$) | 
| $Th$ | Requirement satisfaction threshold values |
| $Sc$ | Requirement satisfaction score values |
| $E =⟨eventType, loc, t_{detect}, [d_{exp},S]⟩$ | Event |
| $t_{detect}$ | Event detection time |
| $d_{exp}$ | Expected event duration |
| $S$ | Event severity |
| $\tau = \langle param, loc, time, r, context \rangle$ | Generic observation task |
| $\tau_{default} = \langle param, loc, [0, \infty), r, O_{ref} \rangle$ | Default observation task | 
| $\tau_{event}=⟨param, loc, time, r, [O_{ref},E]⟩$ | Event-driven observation task |
| $\tau_{event}=⟨param, loc, time, r, \empty⟩$ | Context-free event-driven observation task |
| $loc$ | Desired observation location | 
| $time$ | Task availability time or interval | 
| $r$ | Task priority (or reward) | 
| $O_{ref}$ | Mission objective relevant to a task | 
| $\sigma = \langle \Tau_{parent}, I, \Theta, TW, R_{duration} \rangle$ | Specific observation task |
| $\Tau_{parent}$ | Set of parent generic tasks |
| $I$ | Instrument assigned to perform a specific observation task |
| $\Theta$ | Range of feasible attitude orientations for performing a specific observation task |
| $TW$ | Accessibility time window for a specific observation task to be scheduled |
|$R_{duration}$ | Observation duration requirements for a specific observatio ntask |
| $i$ | Index for satellite $i$ |
| $j$ | Index for generic observation task $j$ |
| $k$ | Index for objective $k$ | 
| $l$ | Index for performance attribute $l$ |
| $n$ | Index for specific observation task $j$ |
| $Rel_{jk}$ | Relevance of task $j$ to objective $k$ |
| $  P_{ijk} = \prod_{l \in R(O_k)}  p_{ikl}(x_{ijl}) $ | Performance of an observation by agent $i$ of task $k$ on objective $k$ |
| $V_{ij} = r_{j} \cdot \sum_{k \in M} w_{ik} \cdot Rel_{jk} \cdot P_{ijk} $ | Value of agent $i$ performing generic task $j$ |
| $U_{in} = \sum_{j \in \Tau_n} V_{ij} - \alpha E_{in} $ | Utility of agent $i$ performing specific task $n$ |
| $E_{in}$ | Cost of agent $i$ for performing specific task $n$ | 
| $\alpha$ | Task cost normalizing parameter | 


<!--

Where:

- 
- 
- 
-   
-  -->