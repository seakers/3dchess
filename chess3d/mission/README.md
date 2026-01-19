# Mission Requirements
## Supported Attributes Parameters
| Attribute | Description |
|-----------|-------------|
| `t_img`   | Imaging time |
| `n_obs`   | Observation number |
| `t_prev`  | Latest prior observation time | 
| `observation range [km]` | Distance between instrument and target point at the time of observation |
| `look angle [deg]` | Angle between nadir and the location of the target relative to the instrument's location |
| `incidence angle [deg]` | Angle between target position and instrument's position relative to the target | 
| `off-nadir axis angle [deg]` | Look angle projection onto the cross-track plane | 
| `ground pixel along-track resolution [m]` | Spatial resolution along-track |  
| `ground pixel cross-track resolution [m]` | Spatial resolution cross-track |

## Mission Requirement Types


## Supported Preference Models
| Strategy | Preference Function | Parameters | Description | 
| ------- | ----------- | ----- |-|
| `categorical` | $ f(x)=  \begin{cases} 1,& \text{if } x\in 1 \\ 0, & \text{otherwise} \end{cases}$  | cat | |