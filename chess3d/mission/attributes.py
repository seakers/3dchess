from enum import Enum

class TemporalRequirementAttributes(Enum):
    DURATION = 'duration [s]'
    REVISIT_TIME = 'revisit_time [s]'
    CO_OBSERVATION_TIME = 'coobservation_time [s]'
    RESPONSE_TIME = 'response_time [s]'
    RESPONSE_TIME_NORM = 'response_time [normalized]'
    OBS_TIME = 't_img [s]'

class ObservationRequirementAttributes(Enum):
    OBSERVATION_NUMBER = 'n_obs'
    SPATIAL_RESOLUTION_CROSS_TRACK = 'ground pixel cross-track resolution [m]'
    SPATIAL_RESOLUTION_ALONG_TRACK = 'ground pixel along-track resolution [m]'
    SPECTRAL_RESOLUTION = 'spectral_resolution'
    ACCURACY = 'accuracy [m]'
    RANGE = 'observation range [km]'
    SNR = 'snr [dB]'
    LOOK_ANGLE = 'look_angle [deg]'
    INCIDENCE_ANGLE = 'incidence_angle [deg]'
    OFF_NADIR_ANGLE = 'off-nadir axis angle [deg]'
    ECLIPSE = 'eclipse'

class SpatialCoverageRequirementAttributes(Enum):
    LOCATION = 'location'

class CapabilityRequirementAttributes(Enum):
    INSTRUMENT = 'instrument'
    # IDEAS:
    # BANDWIDTH = 'bandwidth [nm]'
    # SWATH_WIDTH = 'swath width [km]'
    # DATA_RATE = 'data rate [Mbps]'
    # POINTING_ACCURACY = 'pointing accuracy [deg]'
    # MAX_LOOK_ANGLE = 'max look angle [deg]'
    # MAX_INCIDENCE_ANGLE = 'max incidence angle [deg]'
    # MAX_OFF_NADIR_ANGLE = 'max off-nadir axis angle [deg]'
    # MIN_SNR = 'min snr [dB]'
    # MIN_SPATIAL_RESOLUTION_CROSS_TRACK = 'min ground pixel cross-track resolution [m]'
    # MIN_SPATIAL_RESOLUTION_ALONG_TRACK = 'min ground pixel along-track resolution [m]'
    # MIN_SPECTRAL_RESOLUTION = 'min spectral_resolution'
    # MAX_OBSERVATION_RANGE = 'max observation range [km]'