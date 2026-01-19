import unittest

from chess3d.agents.planning.observations import ObservationOpportunity
from chess3d.agents.planning.tasks import DefaultMissionTask
from chess3d.utils import Interval, print_welcome

class TestObservationOpportunity(unittest.TestCase):
    def test_observation_opportunity(self):
        # Create a observation opportunity
        tasks = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="tasks_001"
        )
        opp = ObservationOpportunity(
            tasks=tasks,
            instrument_name="test_instrument",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        
        self.assertIsInstance(opp, ObservationOpportunity)
        self.assertIn(tasks, opp.tasks)
        self.assertEqual(len(opp.tasks), 1)
        self.assertEqual(opp.instrument_name, "test_instrument")
        self.assertEqual(opp.accessibility.left, 2000.0)
        self.assertEqual(opp.accessibility.right, 3000.0)
        self.assertEqual(opp.min_duration, 0.0)
        self.assertEqual(opp.slew_angles.left, 20.0)
        self.assertEqual(opp.slew_angles.right, 45.0)

        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks="tasks", # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks={"tasks"}, # Invalid parent task
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks=tasks, 
                          instrument_name=1234, # Invalid instrument name
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks=tasks, 
                          instrument_name="test_instrument",
                          accessibility="2000.0-3000.0", # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks=tasks, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks={tasks}, 
                          instrument_name="test_instrument",
                          accessibility=Interval(4000.0,41000.0), # Invalid accessibility
                          min_duration= 0.0,
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks=tasks, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= "0.0", # Invalid duration requirements
                          slew_angles=Interval(20.0, 45.0) )
        self.assertRaises(AssertionError, ObservationOpportunity,
                          tasks=tasks, 
                          instrument_name="test_instrument",
                          accessibility=Interval(2000.0,3000.0),
                          min_duration= 0.0,
                          slew_angles="20.0-45.0" ) # Invalid slew angles
    
    def test_observation_opportunity_copy(self):
        # Create a observation opportunity
        tasks = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="tasks_001"
        )
        opp = ObservationOpportunity(
            tasks=tasks,
            instrument_name="test_instrument",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        
        # Create a copy of the task
        task_copy: ObservationOpportunity = opp.copy()
        self.assertEqual(task_copy.instrument_name, "test_instrument")
        self.assertEqual(task_copy.accessibility.left, 2000.0)
        self.assertEqual(task_copy.accessibility.right, 3000.0)
        self.assertEqual(task_copy.min_duration, 0.0)
        self.assertEqual(task_copy.slew_angles.left, 20.0)
        self.assertEqual(task_copy.slew_angles.right, 45.0)
        self.assertEqual(opp, task_copy)
        self.assertIsNot(opp, task_copy)

    def test_observation_opportunity_can_merge(self):
        # Create a observation opportunity
        task_1 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0
        )
        task_2 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 3),
            mission_duration=3600.0,
        )
        opp_1 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )

        # Cannot merge with parent task
        self.assertRaises(AssertionError, opp_1.can_merge, other=task_1)
        # Cannot merge with invalid overlap toggle
        self.assertRaises(AssertionError, opp_1.can_merge, other=opp_1, must_overlap='False')
        # Cannot merge with invalid maximum duration requirements
        self.assertRaises(AssertionError, opp_1.can_merge, other=opp_1, max_duration='10')
        self.assertRaises(AssertionError, opp_1.can_merge, other=opp_1, max_duration=-10)
        
        # Cannot merge with self
        self.assertFalse(opp_1.can_merge(other=opp_1)) 

        # Cannot merge without overlapping slew angles
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),  
            min_duration=10.0,
            slew_angles=Interval(-45.0, -20.0) # non-overlapping slew angles
        )
        self.assertFalse(opp_1.can_merge(other=opp_2)) 
        
        # Can only merge with non-overlapping access windows if specified
        opp_3 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(3100.0,4000.0), # non-overlapping access window
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertFalse(opp_1.can_merge(other=opp_3,must_overlap=True))
        self.assertTrue(opp_1.can_merge(other=opp_3,must_overlap=False))

        # Can merge with overlapping, non-encompassing access windows
        opp_4 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2500.0,4000.0), # overlapping accessing window
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertTrue(opp_1.can_merge(other=opp_4))

        # Cannot merge with overlapping, non-encompassing access windows with restricted duration requirements
        opp_5 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2950.0,4000.0), # accessing window only allows for 50.0 [s] of joint access
            min_duration=100.0, # minimum duration requirement of 100.0 [s]
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertFalse(opp_1.can_merge(other=opp_5))

        # Can merge with encompassing task
        opp_6 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2500.0,3000.0), # access window fully contained within `opp_1`
            min_duration=10.0,
            slew_angles=Interval(25.0, 50.0) 
        )
        self.assertTrue(opp_1.can_merge(other=opp_6))

        # Cannot merge with encompassing task if duration requirements are not met
        opp_7 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2550.0,3000.0), # access window fully contained within `opp_1`
            min_duration=450.0, # minimum duration requirement of 500.0 [s]
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertFalse(opp_1.can_merge(other=opp_7))

        opp_8 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0), # access window is the same as `opp_1`
            min_duration=1000.0, # minimum duration requirement of exceedes maximum duration requirement
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertFalse(opp_1.can_merge(other=opp_8,max_duration=100.0))

        # Can merge with encompassing tasks if duration requirements are met
        opp_9 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(1000.0,4000.0), # access window fully contains `opp_1`
            min_duration=1000.0, # minimum duration requirement of exceedes `opp_1` maximum duration requirement
            slew_angles=Interval(25.0, 50.0)
        )
        self.assertTrue(opp_1.can_merge(other=opp_9,max_duration=1000.0))

    def test_observation_opportunity_merge(self):
        # Create a observation task
        task_1 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="tasks_001"
        )
        task_2 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 45.0, 1, 1),
            mission_duration=3600.0,
            id="tasks_002"
        )

        opp_1 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        # Cannot merge with parent task
        self.assertRaises(AssertionError, opp_1.merge,other=task_1)
        
        # Can merge with self
        merged_task = opp_1.merge(other=opp_1)
        self.assertIsInstance(merged_task, ObservationOpportunity)

        # Cannot merge with a task with different instrument
        opp_2 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_2", # invalid instrument name
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(20.0, 45.0)
        )
        self.assertRaises(AssertionError, opp_1.merge, other=opp_2)

        # Cannot merge with a task with non-overlapping slew angles
        opp_2 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(100.0,200.0),
            min_duration=10.0,
            slew_angles=Interval(0.0, 10.0) # Non-overlapping slew angles
        )
        self.assertRaises(AssertionError, opp_1.merge, other=opp_2)

        # Cannot merge with task with non-overlapping availability when required
        opp_2 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(300.0,400.0), # accessibility interval contained within `opp_1`
            min_duration=100.0, # Restrictive duration requirements
            slew_angles=Interval(0.0, 10.0)
        )
        self.assertRaises(AssertionError, opp_1.merge, other=opp_2, must_overlap=True)

        # Can merge with non-overlapping tasks when allowed
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(220.0,300.0), # Non-overlapping accessibility
            min_duration=20.0,
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_1.accessibility.right-opp_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, opp_2.accessibility.left+opp_2.min_duration)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_2.accessibility.left - opp_1.accessibility.right + opp_1.min_duration + opp_2.min_duration)
        
        # Merge overlapping task with less restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(150.0,300.0), # proceeding overlapping accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_2.accessibility.left+opp_2.min_duration-opp_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, opp_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_1.min_duration)

        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(50.0, 150.0), # preceding overlapping accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_1.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, opp_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_1.min_duration)

        # Merge overlapping task with more restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(150.0,300.0), # proceeding overlapping accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_2.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, opp_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_2.min_duration)
        
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(50.0, 150.0), # preceding overlapping accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_1.accessibility.left+opp_1.min_duration-opp_2.min_duration)
        self.assertEqual(merged_task.accessibility.right, opp_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_2.min_duration)

        # Merge encompassed task with less restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(125.0, 175.0), # encompassed accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_2.accessibility.left+opp_2.min_duration-opp_1.min_duration)
        self.assertEqual(merged_task.accessibility.right, opp_2.accessibility.right-opp_2.min_duration+opp_1.min_duration)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_1.min_duration)

        # Merge encompassed task with more restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(125.0, 175.0), # encompassed accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_2.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, opp_2.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_2.min_duration)

        # Merge encompassing task with less restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(75.0, 225.0), # encompassing accessibility
            min_duration=5.0, # shorter minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_1.accessibility.left)
        self.assertEqual(merged_task.accessibility.right, opp_1.accessibility.right)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_1.min_duration)

        # Merge encompassing task with more restrictive duration requirements
        opp_2 = ObservationOpportunity(
            tasks=task_2,
            instrument_name="test_instrument_1",
            accessibility=Interval(75.0, 225.0), # encompassing accessibility
            min_duration=20.0, # longer minimum duration
            slew_angles=Interval(20.0, 45.0)
        )
        merged_task = opp_1.merge(other=opp_2)
        self.assertIsInstance(merged_task, ObservationOpportunity)
        self.assertTrue(task_1 in merged_task.tasks)
        self.assertTrue(task_2 in merged_task.tasks)
        self.assertTrue(merged_task.instrument_name == 'test_instrument_1')
        self.assertEqual(merged_task.accessibility.left, opp_1.accessibility.left+opp_1.min_duration-opp_2.min_duration)
        self.assertEqual(merged_task.accessibility.right, opp_1.accessibility.right-opp_1.min_duration+opp_2.min_duration)
        self.assertEqual(merged_task.slew_angles, opp_1.slew_angles.intersection(opp_2.slew_angles))
        self.assertEqual(merged_task.min_duration, opp_2.min_duration)
        
    def test_mutual_exclusivity(self):
        # Create parent tasks
        task_1 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 90.0, 1, 2),
            mission_duration=3600.0,
            id="tasks_001"
        )
        task_2 = DefaultMissionTask(
            parameter="test_parameter",
            location=(45.0, 45.0, 1, 1),
            mission_duration=3600.0,
            id="tasks_002"
        )
        
        # Create observation tasks
        opp_1 = ObservationOpportunity(
            tasks=task_1,
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        opp_2 = ObservationOpportunity(
            tasks=task_2,    # different parent task             
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0),
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        opp_3 = ObservationOpportunity(
            tasks=task_1, # same parent task
            instrument_name="test_instrument_1",
            accessibility=Interval(2000.0,3000.0), 
            min_duration= 0.0,
            slew_angles=Interval(20.0, 45.0)
        )
        opp_4 : ObservationOpportunity = opp_1.merge(opp_2)

        # Check mutual exclusivity
        self.assertFalse(opp_1.is_mutually_exclusive(opp_2))
        self.assertFalse(opp_1.is_mutually_exclusive(opp_3))
        self.assertFalse(opp_2.is_mutually_exclusive(opp_3))
        self.assertTrue(opp_1.is_mutually_exclusive(opp_4))

if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Observation Opportunity Definitions Test')
    
    # run tests
    unittest.main()