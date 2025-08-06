import unittest
import uuid

from chess3d.mission.events import GeophysicalEvent
from chess3d.utils import print_welcome


class TestGeophysicalEvent(unittest.TestCase):
    def test_initialization_valid(self):
        event = GeophysicalEvent('Algal Bloom', 
                                 [
                                    (0.0,0.0,0,0),
                                    (1.0,1.0,0,1)
                                  ], 
                                 1.0, 
                                 0.5, 
                                 1.0
                                )
        self.assertEqual(event.event_type, "algal bloom")
        self.assertEqual(event.location, [(0.0,0.0,0,0), (1.0,1.0,0,1)])
        self.assertEqual(event.t_detect, 1.0)
        self.assertEqual(event.d_exp, 0.5)
        self.assertEqual(event.severity, 1.0)
        self.assertEqual(event.t_start, 1.0)
        self.assertIsInstance(uuid.UUID(event.id), uuid.UUID)

    def test_invalid_event_type(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type=123, severity=0.5, location=[0, 0, 0, 0], t_detect=0, d_exp=1)

    def test_invalid_severity(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity="high", location=[0, 0, 0, 0], t_detect=0, d_exp=1)

    def test_invalid_location_type(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity=0.7, location="bad location", t_detect=0, d_exp=1)

    def test_invalid_location_length(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="flood", severity=0.7, location=[0, 0], t_detect=0, d_exp=1)

    def test_invalid_t_start(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="fire", severity=0.6, location=[1, 2, 3, 4], t_detect="0", d_exp=1)

    def test_invalid_t_end(self):
        with self.assertRaises(AssertionError):
            GeophysicalEvent(event_type="fire", severity=0.6, location=[1, 2, 3, 4], t_detect=0, d_exp="1")

    def test_temporal_status_methods(self):
        event = GeophysicalEvent("flood", [1, 2, 3, 4], 0, 100, 1.0, 100)

        self.assertTrue(event.is_future(50))
        self.assertFalse(event.is_active(50))
        self.assertFalse(event.is_expired(50))

        self.assertTrue(event.is_active(150))
        self.assertFalse(event.is_future(150))
        self.assertFalse(event.is_expired(150))

        self.assertTrue(event.is_expired(250))
        self.assertFalse(event.is_active(250))
        self.assertFalse(event.is_future(250))

    def test_to_dict_and_from_dict(self):
        original = GeophysicalEvent("flood", [0.0, 1.0, 2, 3], 100, 100, 0.9)
        as_dict = original.to_dict()
        recreated = GeophysicalEvent.from_dict(as_dict)

        self.assertIsInstance(recreated, GeophysicalEvent)
        self.assertEqual(original, recreated)

    def test_event_equality_and_hash(self):
        e1 = GeophysicalEvent("fire", [1, 1, 1, 1], 0, 10, 0.6)
        e2 = GeophysicalEvent.from_dict(e1.to_dict())

        self.assertEqual(e1, e2)
        self.assertEqual(hash(e1), hash(e2))

    def test_repr_and_str(self):
        event = GeophysicalEvent("storm", [10, 20, 1, 2], 0, 50, 0.7)
        rep = repr(event)
        s = str(event)

        self.assertIn("storm", rep)
        self.assertIn("storm", s)
        self.assertIn("Severity", s)
        self.assertIn("t_start", rep)


if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Geophysical Event Definition Test')

    # run tests
    unittest.main()