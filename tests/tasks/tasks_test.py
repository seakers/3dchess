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
        self.assertEqual(task.id, "EventObservationTask_test_parameter_1.0_0_1_EVENT-None")
        self.assertRaises(AssertionError, EventObservationTask, parameter="test_parameter", objective=1.0) # wrong type for objective parameter
        self.assertRaises(NotImplementedError, EventObservationTask, parameter="other_parameter", objective=no_target_objective, priority=10) # no location given or specified in objective
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=no_availability_objective, priority=10) # no availability given or specified in objective
        self.assertRaises(AssertionError, EventObservationTask, parameter="other_parameter", objective=objective) # no priority given 

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
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        
        self.assertIsInstance(task, SpecificObservationTask)
        self.assertIn(parent_task, task.parent_tasks)
        self.assertEqual(len(task.parent_tasks), 1)
        self.assertEqual(task.instrument_name, "test_instrument")
        self.assertEqual(task.accessibility.left, 2000.0)
        self.assertEqual(task.accessibility.right, 3000.0)
        self.assertEqual(task.min_duration, 0.0)
        self.assertEqual(task.slew_angles.left, 20.0)
        self.assertEqual(task.slew_angles.right, 45.0)

        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task="parent_task", # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task={"parent_task"}, # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name=1234, # Invalid instrument name
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility="2000.0-3000.0", # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task={parent_task}, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= "0.0", # Invalid duration requirements
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, SpecificObservationTask,
                          parent_task=parent_task, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
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
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        
        # Create a copy of the task
        task_copy: SpecificObservationTask = task.copy()
        self.assertEqual(task_copy.instrument_name, "test_instrument")
        self.assertEqual(task_copy.accessibility.left, 2000.0)
        self.assertEqual(task_copy.accessibility.right, 3000.0)
        self.assertEqual(task_copy.min_duration, 0.0)
        self.assertEqual(task_copy.slew_angles.left, 20.0)
        self.assertEqual(task_copy.slew_angles.right, 45.0)
        self.assertNotEqual(task, task_copy)
    def test_specific_observation_task_can_merge(self):
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
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )

        # Cannot merge with parent task
        self.assertRaises(AssertionError, task_1.can_merge, other_task=parent_task)
        # Cannot merge with invalid overlap toggle
        self.assertRaises(AssertionError, task_1.can_merge, other_task=task_1, must_overlap='False')
        # Cannot merge with invalid maximum duration requirements
        self.assertRaises(AssertionError, task_1.can_merge, other_task=task_1, max_duration='10')
        self.assertRaises(AssertionError, task_1.can_merge, other_task=task_1, max_duration=-10)
        
        # Can merge with self
        self.assertTrue(task_1.can_merge(other_task=task_1)) 

        # Cannot merge without overlapping slew angles
        task_2 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),  
            min_duration=10.0,
            slew_angles=Interval(-45.0, -20.0) # non-overlapping slew angles
        )
        self.assertFalse(task_1.can_merge(other_task=task_2)) 
        
        # Can only merge with non-overlapping access windows if specified
        task_3 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(3100.0,4000.0), # non-overlapping access window
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertFalse(task_1.can_merge(other_task=task_3,must_overlap=True))
        self.assertTrue(task_1.can_merge(other_task=task_3,must_overlap=False))

        # Can merge with overlapping, non-encompassing access windows
        task_4 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2500.0,4000.0), # overlapping accessing window
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertTrue(task_1.can_merge(other_task=task_4))

        # Cannot merge with overlapping, non-encompassing access windows with restricted duration requirements
        task_5 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2950.0,4000.0), # accessing window only allows for 50.0 [s] of joint access
            min_duration=100.0, # minimum duration requirement of 100.0 [s]
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertFalse(task_1.can_merge(other_task=task_5))

        # Can merge with encompassing task
        task_6 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2500.0,3000.0), # access window fully contained within `task_1`
            min_duration=10.0,
            slew_angles=Interval(25.0, 50.0) 
        )
        self.assertTrue(task_1.can_merge(other_task=task_6))

        # Cannot merge with encompassing task if duration requirements are not met
        task_7 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2550.0,3000.0), # access window fully contained within `task_1`
            min_duration=450.0, # minimum duration requirement of 500.0 [s]
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertFalse(task_1.can_merge(other_task=task_7))

        task_8 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0), # access window is the same as `task_1`
            min_duration=1000.0, # minimum duration requirement of exceedes maximum duration requirement
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertFalse(task_1.can_merge(other_task=task_8,max_duration=100.0))

        # Can merge with encompassing tasks if duration requirements are met
        task_9 = SpecificObservationTask(
            parent_task=parent_task,
            instrument_name="test_instrument_1",
            accessibility=Interval(1000.0,4000.0), # access window fully contains `task_1`
            min_duration=1000.0, # minimum duration requirement of exceedes `task_1` maximum duration requirement
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertTrue(task_1.can_merge(other_task=task_9,max_duration=1000.0))


    def test_specific_observation_task_merge(self):
        # Create a specific observation task
        parent_task_1 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        parent_task_2 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 45.0, 1, 1),
            mission_duration=3600.0,
            id="parent_task_002"
        )

        task_1 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        # Cannot merge with parent task
        self.assertRaises(AssertionError, task_1.merge,other_task=parent_task_1)
        
        # Can merge with self
        merged_task = task_1.merge(other_task=task_1)
        self.assertIsInstance(merged_task, SpecificObservationTask)

        # Cannot merge with a task with different instrument
        task_2 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_2", # invalid instrument name
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertRaises(AssertionError, task_1.merge, other_task=task_2)

        # Cannot merge with a task with non-overlapping slew angles
        task_2 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(0.0, 10.0) # Non-overlapping slew angles
        )
        self.assertRaises(AssertionError, task_1.merge, other_task=task_2)

        # Cannot merge with task with non-overlapping availability when required
        task_2 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(300.0,400.0), # accessibility interval contained within `task_1`
            min_duration=100.0, # Restrictive duration requirements
            slew_angles=Interval(0.0, 10.0)
        )
        self.assertRaises(AssertionError, task_1.merge, other_task=task_2, must_overlap=True)

        # Can merge with non-overlapping tasks when allowed
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(220.0,300.0), # Non-overlapping accessibility
            min_duration=20.0,
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_1.accessibility.right-task_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, task_2.accessibility.left+task_2.min_duration)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_2.accessibility.left - task_1.accessibility.right + task_1.min_duration + task_2.min_duration)
        
        # Merge overlapping task with less restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(150.0,300.0), # proceeding overlapping accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_2.accessibility.left+task_2.min_duration-task_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, task_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_1.min_duration)

        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(50.0, 150.0), # preceding overlapping accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_1.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, task_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_1.min_duration)

        # Merge overlapping task with more restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(150.0,300.0), # proceeding overlapping accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_2.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, task_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_2.min_duration)
        
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(50.0, 150.0), # preceding overlapping accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_1.accessibility.left+task_1.min_duration-task_2.min_duration)
        self.assertEqual(merged_task.accessibility.right, task_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_2.min_duration)

        # Merge encompassed task with less restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(125.0, 175.0), # encompassed accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_2.accessibility.left+task_2.min_duration-task_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, task_2.accessibility.right-task_2.min_duration+task_1.min_duration)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_1.min_duration)

        # Merge encompassed task with more restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(125.0, 175.0), # encompassed accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_2.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, task_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_2.min_duration)

        # Merge encompassing task with less restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(75.0, 225.0), # encompassing accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_1.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, task_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_1.min_duration)

        # Merge encompassing task with more restrictive duration requirements
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(75.0, 225.0), # encompassing accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = task_1.merge(other_task=task_2)
        self.assertIsInstance(merged_task, SpecificObservationTask)
        self.assertTrue(parent_task_1 in merged_task.parent_tasks)
        self.assertTrue(parent_task_2 in merged_task.parent_tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, task_1.accessibility.left+task_1.min_duration-task_2.min_duration)
        self.assertEqual(merged_task.accessibility.right, task_1.accessibility.right-task_1.min_duration+task_2.min_duration)
        self.assertEqual(merged_task.slew_angles, task_1.slew_angles.intersection(task_2.slew_angles))
        self.assertEqual(merged_task.min_duration, task_2.min_duration)
        
    def test_mutual_exclusivity(self):
        # Create parent tasks
        parent_task_1 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="parent_task_001"
        )
        parent_task_2 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 45.0, 1, 1),
            mission_duration=3600.0,
            id="parent_task_002"
        )
        
        # Create specific observation tasks
        task_1 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        task_2 = SpecificObservationTask(
            parent_task=parent_task_2,                  
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        task_3 = SpecificObservationTask(
            parent_task=parent_task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        task_4 : SpecificObservationTask = task_1.merge(task_2)

        # Check mutual exclusivity
        self.assertFalse(task_1.is_mutually_exclusive(task_2))
        self.assertTrue(task_1.is_mutually_exclusive(task_3))
        self.assertFalse(task_2.is_mutually_exclusive(task_3))
        self.assertTrue(task_1.is_mutually_exclusive(task_4))

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Definitions Test')
    
    # run tests
    unittest.main()