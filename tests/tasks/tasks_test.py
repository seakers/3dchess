import unittest

from chess3d.agents.planning.tasks import *
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
    def test_default_mission_task_generate_id(self):
        # Create a default mission task
        task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0
        )
        # Generate ID
        self.assertEqual(task.generate_id(), "GenericObservation_test_parameter_1.0_1_2")
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
        self.assertNotEqual(task,task_copy)
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
        self.assertEqual(repr(task), "DefaultMissionTask(parameter=test_parameter, priority=1.0, location=[(45.0, 90.0, 1, 2)], availability=[0.0,3600.0], id=test_task_001)")

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
        self.assertEqual(task.id, "EventObservationTask_test_parameter_1.0_0_1_EVENT-None")
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
        self.assertEqual(task.id, f"EventObservationTask_test_parameter_{event.severity}_0_1_EVENT-{event.id.split('-')[0]}")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=1.0)
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=event, location=[(45.0, 90.0, 0, 1)])
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=event, availability=Interval(0.0, 3600.0))
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", event=event, priority=1.0)
    def test_event_driven_task_no_event_with_objective(self):
        # Create a mission objective
        objective = DefaultMissionObjective(
            parameter="test_parameter",
            weight=1.0
        )

        task = EventObservationTask(
            parameter="test_parameter",
            location=[(45.0, 90.0, 0, 1)],
            availability=Interval(0.0, 3600.0),
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
        self.assertEqual(task.id, "EventObservationTask_test_parameter_1.0_0_1_EVENT-None")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", objective=1.0)
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=objective)
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
        self.assertEqual(task_copy.id, f"EventObservationTask_test_parameter_{event.severity}_0_1_EVENT-{event.id.split('-')[0]}")
        self.assertNotEqual(task, task_copy)
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
        self.assertEqual(task_dict['id'], f"EventObservationTask_test_parameter_{event.severity}_0_1_EVENT-{event.id.split('-')[0]}")
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
            'availability': {'left': 1000.0, 'right': 4600.0},
            'id': f"EventObservationTask_test_parameter_5.0_0_1_EVENT-{event_dict['event_type']}"
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
        self.assertEqual(task.id, f"EventObservationTask_test_parameter_{event_dict['severity']}_0_1_EVENT-{event_dict['event_type']}")
    
    # TODO Specific Observation Task Tests
    def test_specific_observation_task(self):
        # Create a specific observation task
        parent_task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        task = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument",
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        
        self.assertIsInstance(task, SpecificObservationTask)
        self.assertIn(parent_task, task.parent_tasks)
        self.assertEqual(len(task.parent_tasks), 1)
        self.assertEqual(task.instrument_name, "test_instrument")
        self.assertEqual(task.accessibility.left, 2000.0)
        self.assertEqual(task.accessibility.right, 3000.0)
        self.assertEqual(task.duration_requirements.left, 0.0)
        self.assertEqual(task.duration_requirements.right, 200.0)
        self.assertEqual(task.slew_angles.left, 20.0)
        self.assertEqual(task.slew_angles.right, 45.0)

        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task="parent_task", # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task={"parent_task"}, # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name=1234, # Invalid instrument name
                          accessibility=Interval(2000.0,3000.0),
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility="2000.0-3000.0", # Invalid accessibility
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task={parent_task}, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          duration_requirements="0.0-200.0", # Invalid duration requirements
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          duration_requirements= Interval(0.0, 200.0),
                          slew_angles="20.0-45.0" ) # Invalid slew angles
    def test_specific_observation_task_copy(self):
        # Create a specific observation task
        parent_task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        task = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument",
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        
        # Create a copy of the task
        task_copy: SpecificObservationTask = task.copy()
        self.assertEqual(task_copy.instrument_name, "test_instrument")
        self.assertEqual(task_copy.accessibility.left, 2000.0)
        self.assertEqual(task_copy.accessibility.right, 3000.0)
        self.assertEqual(task_copy.duration_requirements.left, 0.0)
        self.assertEqual(task_copy.duration_requirements.right, 200.0)
        self.assertEqual(task_copy.slew_angles.left, 20.0)
        self.assertEqual(task_copy.slew_angles.right, 45.0)
        self.assertNotEqual(task, task_copy)
    def test_specific_observation_task_can_combine(self):
        # Create a specific observation task
        parent_task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        task_1 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        task_2 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_2", # invalid instrument name
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        task_3 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0), 
            duration_requirements= Interval(0.0, 200.0), 
            slew_angles=Interval(0.0, 10.0) # Non-overlapping slew angles
        )
        task_4 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(1000.0,2500.0),
            duration_requirements= Interval(100.0, 1000.0), # Restrictive duration requirements
            slew_angles=Interval(0.0, 10.0)
        )
        task_5 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(0.0,1000.0), # Non-overlapping accessibility
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )

        self.assertRaises(ValueError, task_1.can_combine, other_task=parent_task)
        self.assertTrue(task_1.can_combine(other_task=task_1))
        self.assertFalse(task_1.can_combine(other_task=task_2))
        self.assertFalse(task_1.can_combine(other_task=task_3))
        self.assertFalse(task_1.can_combine(other_task=task_4))
        self.assertFalse(task_1.can_combine(other_task=task_5,extend=False))
    def test_specific_observation_task_merge(self):
        # Create a specific observation task
        parent_task = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        task_1 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        task_2 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_2", # invalid instrument name
            accessibility=Interval(2000.0,3000.0),
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        task_3 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0), 
            duration_requirements= Interval(0.0, 200.0), 
            slew_angles=Interval(0.0, 10.0) # Non-overlapping slew angles
        )
        task_4 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(1000.0,2500.0),
            duration_requirements= Interval(100.0, 1000.0), # Restrictive duration requirements
            slew_angles=Interval(0.0, 10.0)
        )
        task_5 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(0.0,1000.0), # Non-overlapping accessibility
            duration_requirements= Interval(0.0, 200.0),
            slew_angles=Interval(20.0, 45.0)
        )
        task_6 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(1000.0,3500.0),
            duration_requirements= Interval(100.0, 150.0),
            slew_angles=Interval(10.0, 35.0)
        )

        self.assertRaises(ValueError, task_1.merge, other_task=parent_task)
        self.assertRaises(AssertionError, task_1.merge, other_task=task_2)
        self.assertRaises(AssertionError, task_1.merge, other_task=task_3)
        self.assertRaises(AssertionError, task_1.merge, other_task=task_4)
        self.assertRaises(AssertionError, task_1.merge, other_task=task_5, extend=False)
        
        merged_task : SpecificObservationTask = task_1.merge(other_task=task_6)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertIn(parent_task, merged_task.parent_tasks)
        self.assertEqual(len(merged_task.parent_tasks), 1)
        self.assertEqual(merged_task.instrument_name, "test_instrument_1")
        self.assertEqual(merged_task.accessibility.left, 1000.0)
        self.assertEqual(merged_task.accessibility.right, 3500.0)
        self.assertEqual(merged_task.duration_requirements.left, 100.0)
        self.assertEqual(merged_task.duration_requirements.right, 150.0)
        self.assertEqual(merged_task.slew_angles.left, 20.0)
        self.assertEqual(merged_task.slew_angles.right, 35.0)
        
if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Definitions Test')
    
    # run tests
    unittest.main()