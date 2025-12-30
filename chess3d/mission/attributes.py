from enum import Enum

class TemporalRequirementAttributes(Enum):
    DURATION = 'duration'
    REVISIT_TIME = 'revisit_time'
    RELATIVE_OBS_TIME = 'observation_time'
    OBS_TIME = 't_img'

class ObservationRequirementAttributes(Enum):
    OBSERVATION_NUMBER = 'n_obs'
    SPATIAL_RESOLUTION_CROSS_TRACK = 'ground pixel cross-track resolution [m]'
    SPATIAL_RESOLUTION_ALONG_TRACK = 'ground pixel along-track resolution [m]'
    SPECTRAL_RESOLUTION = 'spectral_resolution'
    RANGE = 'observation range [km]'
    SNR = 'snr'
    LOOK_ANGLE = 'look_angle [deg]'
    INCIDENCE_ANGLE = 'incidence_angle [deg]'
    OFF_NADIR_ANGLE = 'off-nadir axis angle [deg]'

class SpatialCoverageRequirementAttributes(Enum):
    LOCATION = 'location'