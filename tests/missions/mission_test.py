import unittest

from chess3d.mission.events import GeophysicalEvent
from chess3d.mission.mission import *
from chess3d.mission.requirements import *
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome


class TestMission(unittest.TestCase):
    def setUp(self):
        reqs_1 = [
            ContinuousRequirement(
                attribute="horizontal_spatial_resolution",
                thresholds=[10, 30, 100],
                scores=[1.0, 0.7, 0.1]
            ),
            CategoricalRequirement(
                attribute="spectral_resolution",
                thresholds=["Hyperspectral", "Multispectral"],
                scores=[1, 0.5]
            ),
            CapabilityRequirement(
                attribute="instrument",
                valid_values=["VNIR", "TIR"]
            ),
            RevisitTemporalRequirement(
                thresholds=[3600, 14400, 86400],
                scores=[1.0, 0.5, 0.0]
            ),
            GridTargetSpatialRequirement(
                grid_name="grid_1",
                grid_index=0,
                grid_size=10
            )
        ]

        reqs_2 = [
            ContinuousRequirement(
                attribute="horizontal_spatial_resolution",
                thresholds=[30, 100],
                scores=[1.0, 0.3]
            ),
            CapabilityRequirement(
                attribute="instrument",
                valid_values=["TIR"]
            ),
            RevisitTemporalRequirement(
                thresholds=[3600, 14400, 86400],
                scores=[1.0, 0.5, 0.0]
            ),
            GridTargetSpatialRequirement(
                grid_name="grid_1",
                grid_index=0,
                grid_size=10
            )
        ]

        reqs_3 = [
            ContinuousRequirement(
                attribute="horizontal_spatial_resolution",
                thresholds=[30, 100],
                scores=[1.0, 0.5]
            ),
            ContinuousRequirement(
                attribute="accuracy",
                thresholds=[1, 5, 10],
                scores=[1.0, 0.5, 0.1]
            ),
            CapabilityRequirement(
                attribute="instrument",
                valid_values=["Altimeter"]
            ),
            RevisitTemporalRequirement(
                thresholds=[3600, 14400, 86400],
                scores=[1.0, 0.5, 0.0]
            ),
            GridTargetSpatialRequirement(
                grid_name="grid_1",
                grid_index=0,
                grid_size=10
            )
        ]
        
        default_objective_1 = DefaultMissionObjective(
            parameter="Chlorophyll-A",
            priority=1,
            requirements=reqs_1
        )

        default_objective_2 = DefaultMissionObjective(
            parameter="Water temperature",
            priority=1,
            requirements=reqs_2
        )

        default_objective_3 = DefaultMissionObjective(
            parameter="Water level",
            priority=1,
            requirements=reqs_3
        )

        event_objective_1 = EventDrivenObjective(
            event_type="algal bloom",
            parameter="Chlorophyll-A",
            priority=10,
            requirements=reqs_1,
            synergistic_parameters=["Water temperature", "Water level"]
        )

        event_objective_2 = EventDrivenObjective(
            event_type="algal bloom",
            parameter="Water temperature",
            priority=10,
            requirements=reqs_2,
            synergistic_parameters=["Chlorophyll-A", "Water level"]
        )

        event_objective_3 = EventDrivenObjective(
            event_type="algal bloom",
            parameter="Water level",
            priority=10,
            requirements=reqs_3,
            synergistic_parameters=["Water temperature", "Chlorophyll-A"]
        )

        self.objectives = [
            default_objective_1,
            default_objective_2,
            default_objective_3,
            event_objective_1,
            event_objective_2,
            event_objective_3
        ]
    
    def test_mission_initialization(self):
        ...

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Mission Definitions Test')
    
    # run tests
    unittest.main()