import unittest

from chess3d.agents.planning.tasks import *
from chess3d.agents.science.requests import TaskRequest
from chess3d.mission.objectives import *
from chess3d.utils import print_welcome


class TestGenericTasks(unittest.TestCase):
    # Default Mission Task Tests
    def test_default_mission_task(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,            
            id="test_task_001"
        )
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.location[0], (45.0, 90.0, 1, 2))
        self.assertEqual(task.availability.right, 3600.0)
        self.assertEqual(task.priority, 1.0)
        self.assertEqual(task.id, "test_task_001")
    
    def test_default_mission_task_copy(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="test_task_001"
        )
        # Create a copy of the task
        task_copy : DefaultMissionTask = task.copy()
        self.assertEqual(task_copy.parameter, "test_parameter")
        self.assertEqual(task_copy.location[0], (45.0, 90.0, 1, 2))
        self.assertEqual(task_copy.availability.right, 3600.0)
        self.assertEqual(task.to_dict(), task_copy.to_dict())
        self.assertIsNot(task, task_copy)

    def test_default_mission_task_availability(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="test_task_001"
        )
        # Check availability
        self.assertTrue(task.is_available(1800.0))
        self.assertFalse(task.is_available(4000.0))
        self.assertRaises(AssertionError, task.is_available, -100.0)

    def test_default_mission_task_to_dict(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="test_task_001"
        )
        # Convert to dict
        task_dict = task.to_dict()
        self.assertEqual(task_dict['parameter'], "test_parameter")
        self.assertEqual(task_dict['location'][0], (45.0, 90.0, 1, 2))
        self.assertEqual(task_dict['availability']['right'], 3600.0)

    def test_default_mission_task_from_dict(self):
        # Create a default mission task
        task_dict = {
            'task_type': GenericObservationTask.DEFAULT,
            'parameter': 'test_parameter',
            'location': [(45.0, 90.0, 1, 2)],
            'availability': {'left': 0.0, 'right': 3600.0}
        }
        task = GenericObservationTask.from_dict(task_dict)
        self.assertIsInstance(task, DefaultMissionTask)
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.location[0], (45.0, 90.0, 1, 2))
        self.assertEqual(task.availability.right, 3600.0)

    def test_default_mission_task_repr(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
        )
        # Check string representation
        self.assertEqual(repr(task), "DefaultMissionTask-'test_parameter'@(1,2)")

    # Event Driven Objective Tests
    def test_event_driven_task_no_event_no_objective(self):
        task = EventObservationTask(
            parameter="test_parameter",
            location=[(45.0, 90.0, 0, 1)],
            availability=Interval(0.0, 3600.0),
            priority=1.0
        )
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.task_type, GenericObservationTask.EVENT)
        self.assertEqual(task.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task.availability.left, 0.0)
        self.assertEqual(task.availability.right, 3600.0)
        self.assertEqual(task.priority, 1.0)
        self.assertIsNone(task.event)
        self.assertIsNone(task.objective)
        self.assertEqual(task.id, "EventObservationTask-'test_parameter'@(0,1)-EVENT-None")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", location=[(45.0, 90.0, 0, 1)])
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", location=[(45.0, 90.0, 0, 1)], priority=1.0)
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", location=[(45.0, 90.0, 0, 1)], availability=Interval(0.0, 3600.0))

    def test_event_driven_task_with_event_no_objective(self):
        # Create a geophysical event
        event = GeophysicalEvent(
            event_type="earthquake",
            severity=5.0,
            location=[(45.0, 90.0, 0, 1)],
            t_detect=1000.0,
            d_exp=3600.0
        )
        task = EventObservationTask(
            parameter="test_parameter",
            event=event
        )

        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.task_type, GenericObservationTask.EVENT)
        self.assertEqual(task.event, event)
        self.assertEqual(task.location[0], event.location[0])
        self.assertEqual(task.availability.left, event.t_start)
        self.assertEqual(task.availability.right, event.t_start + event.d_exp)
        self.assertEqual(task.priority, event.severity)
        self.assertEqual(task.id, f"EventObservationTask-'test_parameter'@(0,1)-EVENT-{event.id.split('-')[0]}")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=1.0)
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter") # no event, objective, or task information specified
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", availability=Interval(0.0, 3600.0), priority=10) # no event, objective, or task location specified
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", location=[(45.0, 90.0, 0, 1)], priority=10) # no event, objective, or task availability specified
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", location=[(45.0, 90.0, 0, 1)], availability=Interval(0.0, 3600.0)) # no event, objective, or task priority specified
    
    def test_event_driven_task_no_event_with_objective(self):
        # Create a mission objective
        objective = EventDrivenObjective(
            event_type="flood",
            parameter="test_parameter",
            weight=1.0,
            requirements=[
                # Define any specific requirements for the objective here
                PointTargetSpatialRequirement((45.0, 90.0, 0, 1)),
                AvailabilityRequirement(0, 3600.0),
            ],
        )
        no_target_objective = EventDrivenObjective(
            event_type="flood",
            parameter="test_parameter",
            weight=1.0,
            requirements=[
                AvailabilityRequirement(0, 3600.0),
            ],
        )
        no_availability_objective = EventDrivenObjective(
            event_type="flood",
            parameter="test_parameter",
            weight=1.0,
            requirements=[
                PointTargetSpatialRequirement((45.0, 90.0, 0, 1))
            ],
        )
        task = EventObservationTask(
            parameter="test_parameter",
            priority=1.0,
            objective=objective
        )
        self.assertEqual(task.parameter, objective.parameter)
        self.assertEqual(task.task_type, GenericObservationTask.EVENT)
        self.assertIsNone(task.event)
        self.assertEqual(task.objective, objective)
        self.assertEqual(task.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task.availability.left, 0.0)
        self.assertEqual(task.availability.right, 3600.0)
        self.assertEqual(task.priority, 1.0)
        self.assertEqual(task.id, "EventObservationTask-'test_parameter'@(0,1)-EVENT-None")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", objective=1.0) # wrong type for objective parameter
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=no_target_objective, priority=10) # no location given or specified in objective
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=no_availability_objective, priority=10) # no availability given or specified in objective
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=objective) # no priority given 

    def test_event_driven_task_with_event_with_objective(self):
        # Create a geophysical event
        event = GeophysicalEvent(
            event_type="earthquake",
            severity=5.0,
            location=[(45.0, 90.0, 0, 1)],
            t_detect=1000.0,
            d_exp=3600.0
        )
        
        # Create a mission objective
        objective = EventDrivenObjective(
            event_type="earthquake",
            parameter="test_parameter",
            weight=1.0,
            requirements=[
                PointTargetSpatialRequirement((45.0, 90.0, 0, 1)),
                AvailabilityRequirement(0, 3600.0),
            ],
        )
        wrong_objective = EventDrivenObjective(
            event_type="flood",
            parameter="test_parameter",
            weight=1.0,
            requirements=[
                PointTargetSpatialRequirement((45.0, 90.0, 0, 1)),
                AvailabilityRequirement(0, 3600.0),
            ]
        )

        # Check initialization
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=event, objective=wrong_objective) # event type mismatch between event and objective

        # Create task
        task = EventObservationTask(
            parameter="test_parameter",
            priority=1.0,
            objective=objective,
            event=event
        )
        self.assertEqual(task.parameter, objective.parameter)
        self.assertEqual(task.task_type, GenericObservationTask.EVENT)
        self.assertEqual(task.event, event)

    def test_event_driven_objective_copy(self):
        # Create a geophysical event
        event = GeophysicalEvent(
            event_type="earthquake",
            severity=5.0,
            location=[(45.0, 90.0, 0, 1)],
            t_detect=1000.0,
            d_exp=3600.0
        )

        task = EventObservationTask(
            parameter="test_parameter",
            event=event
        )
        # Create a copy of the task
        task_copy: EventObservationTask = task.copy()
        self.assertEqual(task_copy.parameter, "test_parameter")
        self.assertEqual(task_copy.task_type, GenericObservationTask.EVENT)
        self.assertEqual(task_copy.event, event)
        self.assertEqual(task_copy.location[0], event.location[0])
        self.assertEqual(task_copy.availability.left, event.t_start)
        self.assertEqual(task_copy.availability.right, event.t_start + event.d_exp)
        self.assertEqual(task_copy.priority, event.severity)
        self.assertEqual(task_copy.id, f"EventObservationTask-'test_parameter'@(0,1)-EVENT-{event.id.split('-')[0]}")
        self.assertEqual(task.to_dict(), task_copy.to_dict())
        self.assertIsNot(task, task_copy)

    def test_event_driven_objective_availability(self):
        # Create a geophysical event
        event = GeophysicalEvent(
            event_type="earthquake",
            severity=5.0,
            location=[(45.0, 90.0, 0, 1)],
            t_detect=1000.0,
            d_exp=3600.0
        )

        task = EventObservationTask(
            parameter="test_parameter",
            event=event
        )
        # Check availability
        self.assertTrue(task.is_available(1800.0))
        self.assertFalse(task.is_available(5000.0))
        self.assertRaises(AssertionError, task.is_available, -100.0)

    def test_event_driven_objective_to_dict(self):
        # Create a geophysical event
        event = GeophysicalEvent(
            event_type="earthquake",
            severity=5.0,
            location=[(45.0, 90.0, 0, 1)],
            t_detect=1000.0,
            d_exp=3600.0
        )

        task = EventObservationTask(
            parameter="test_parameter",
            event=event
        )
        # Convert to dict
        task_dict = task.to_dict()
        self.assertEqual(task_dict['parameter'], "test_parameter")
        self.assertEqual(task_dict['event']['event_type'], "earthquake")
        self.assertEqual(task_dict['priority'], 5.0)
        self.assertEqual(task_dict['location'][0], (45.0, 90.0, 0, 1))
        self.assertEqual(task_dict['availability']['left'], 1000.0)
        self.assertEqual(task_dict['availability']['right'], 4600.0)
        self.assertEqual(task_dict['id'], f"EventObservationTask-'test_parameter'@(0,1)-EVENT-{event.id.split('-')[0]}")

    def test_event_driven_objective_from_dict(self):
        # Create a geophysical event
        event_dict = {
            'event_type': 'earthquake',
            'severity': 5.0,
            'location': [(45.0, 90.0, 0, 1)],
            't_detect': 1000.0,
            'd_exp': 3600.0
        }

        task_dict = {
            'task_type': GenericObservationTask.EVENT,
            'parameter': 'test_parameter',
            'event': event_dict,
            'availability': {'left': 1000.0, 'right': 4600.0}
        }
        task = EventObservationTask.from_dict(task_dict)
        self.assertIsInstance(task, EventObservationTask)
        self.assertEqual(task.task_type, GenericObservationTask.EVENT)
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.location, [(45.0, 90.0, 0, 1)])
        self.assertEqual(task.availability.left, 1000.0)
        self.assertEqual(task.availability.right, 4600.0)
        self.assertEqual(task.priority, 5.0)
        self.assertEqual(task.event.event_type, "earthquake")
        self.assertEqual(task.event.severity, 5.0)
        self.assertEqual(task.event.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task.event.t_detect, 1000.0)
        self.assertEqual(task.event.d_exp, 3600.0)
        self.assertEqual(task.objective, None)
        self.assertEqual(task.id, f"EventObservationTask-'test_parameter'@(0,1)-EVENT-{task.event.id.split('-')[0]}")

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Definitions Test')
    
    # run tests
    unittest.main()