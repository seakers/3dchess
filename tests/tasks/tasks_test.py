import unittest

from chess3d.agents.planning.tasks import *
from chess3d.mission.mission import *
from chess3d.mission.requirements import *
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
        self.assertEqual(task.id, "test_task_001")
        self.assertEqual(task.generate_id(), "GenericObservation_test_parameter_1_2")
    def test_default_mission_task_generate_id(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="test_task_001"
        )
        # Generate ID
        generated_id = task.generate_id()
        self.assertEqual(generated_id, "GenericObservation_test_parameter_1_2")
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
        self.assertEqual(task_copy.id, "test_task_001")
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
        self.assertEqual(task_dict['id'], "test_task_001")
    def test_default_mission_task_from_dict(self):
        # Create a default mission task
        task_dict = {
            'task_type': GenericObservationTask.DEFAULT,
            'parameter': 'test_parameter',
            'location': [(45.0, 90.0, 1, 2)],
            'availability': {'left': 0.0, 'right': 3600.0},
            'id': 'test_task_001'
        }
        task = GenericObservationTask.from_dict(task_dict)
        self.assertIsInstance(task, DefaultMissionTask)
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.location[0], (45.0, 90.0, 1, 2))
        self.assertEqual(task.availability.right, 3600.0)
        self.assertEqual(task.id, "test_task_001")
    def test_default_mission_task_repr(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="test_task_001"
        )
        # Check string representation
        self.assertEqual(repr(task), "DefaultMissionTask(parameter=test_parameter, location=[(45.0, 90.0, 1, 2)], availability=[0.0,3600.0], id=test_task_001)")

    # Event Driven Objective Tests
    def test_event_driven_task(self):
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
        self.assertEqual(task.event.event_type, "earthquake")
        self.assertEqual(task.event.severity, 5.0)
        self.assertEqual(task.event.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task.event.t_detect, 1000.0)
        self.assertEqual(task.event.d_exp, 3600.0)
        self.assertEqual(task.generate_id(), f"EventObservationTask_test_parameter_0_1_EVENT-{event.id.split('-')[0]}")
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
        self.assertEqual(task_copy.event.event_type, "earthquake")
        self.assertEqual(task_copy.event.severity, 5.0)
        self.assertEqual(task_copy.event.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task_copy.event.t_detect, 1000.0)
        self.assertEqual(task_copy.event.d_exp, 3600.0)
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
        self.assertFalse(task.is_available(500.0))
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
        self.assertEqual(task_dict['event']['severity'], 5.0)
        self.assertEqual(task_dict['event']['location'][0], (45.0, 90.0, 0, 1))
        self.assertEqual(task_dict['event']['t_detect'], 1000.0)
        self.assertEqual(task_dict['event']['d_exp'], 3600.0)
        self.assertEqual(task_dict['id'], f"EventObservationTask_test_parameter_0_1_EVENT-{event.id.split('-')[0]}")
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
            'id': f"EventObservationTask_test_parameter_0_1_EVENT-{event_dict['event_type']}"
        }
        task = EventObservationTask.from_dict(task_dict)
        self.assertIsInstance(task, EventObservationTask)
        self.assertEqual(task.parameter, "test_parameter")
        self.assertEqual(task.event.event_type, "earthquake")
        self.assertEqual(task.event.severity, 5.0)
        self.assertEqual(task.event.location[0], (45.0, 90.0, 0, 1))
        self.assertEqual(task.event.t_detect, 1000.0)
        self.assertEqual(task.event.d_exp, 3600.0)
        self.assertEqual(task.id, f"EventObservationTask_test_parameter_0_1_EVENT-{event_dict['event_type']}")
    def test_event_driven_objective_repr(self):
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
        # Check string representation
        self.assertEqual(repr(task), f"EventObservationTask(parameter=test_parameter, event={event}, location={event.location}, availability={task.availability}, id=EventObservationTask_test_parameter_0_1_EVENT-{event.id.split('-')[0]})")

    # TODO Specific Observation Task Tests

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Definitions Test')
    
    # run tests
    unittest.main()