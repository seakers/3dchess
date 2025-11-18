
import unittest

from chess3d.utils import print_welcome
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.planning.tasks import DefaultMissionTask


class TestBids(unittest.TestCase):
    """
    # Test suite for bids used in consensus planners.
    """

    def setUp(self):
        # Common task stub
        self.task = DefaultMissionTask(
            parameter='test-parameter',
            location=(0.0, 0.0, 1, 2),
            mission_duration=1000.0,
            priority=1.0
        )

        # Two basic bids on the same task
        self.bid_a = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-A",
            bid_value=10.0,
            winning_bidder="sat-A",
            winning_bid=10.0,
            t_img=100.0,
            t_update=1.0,
            performed=False,
        )

        self.bid_b = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-B",
            bid_value=5.0,
            winning_bidder="sat-B",
            winning_bid=5.0,
            t_img=110.0,
            t_update=1.0,
            performed=False,
        )

        self.bid_c = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-C",
            bid_value=5.0,
            winning_bidder="sat-C",
            winning_bid=5.0,
            t_img=110.0,
            t_update=1.0,
            performed=False,
        )

    # ---------- Basic construction / serialization ----------

    def test_to_dict_and_from_dict_roundtrip(self):
        """Bid.to_dict() and Bid.from_dict() should be inverse operations."""
        d = self.bid_a.to_dict()
        bid_copy = Bid.from_dict(d)

        self.assertEqual(bid_copy.task, self.bid_a.task)
        self.assertEqual(bid_copy.main_measurement, self.bid_a.main_measurement)
        self.assertEqual(bid_copy.bidder, self.bid_a.bidder)
        self.assertEqual(bid_copy.bid_value, self.bid_a.bid_value)
        self.assertEqual(bid_copy.winning_bidder, self.bid_a.winning_bidder)
        self.assertEqual(bid_copy.winning_bid, self.bid_a.winning_bid)
        self.assertEqual(bid_copy.t_img, self.bid_a.t_img)
        self.assertEqual(bid_copy.t_update, self.bid_a.t_update)
        self.assertEqual(bid_copy.performed, self.bid_a.performed)

    def test_copy_creates_independent_instance(self):
        """copy() should produce a deep copy that can be changed independently."""
        bid_copy = self.bid_a.copy()
        self.assertIsNot(bid_copy, self.bid_a)
        self.assertEqual(bid_copy, self.bid_a)

        # mutate copy and ensure original unchanged
        bid_copy.winning_bid = 123.0
        self.assertNotEqual(bid_copy.winning_bid, self.bid_a.winning_bid)

    # ---------- Comparison operators ----------

    def test_comparison_by_bid_value(self):
        """__lt__ and __gt__ should compare based on bid_value (with tie-breaking)."""
        # bid_a has higher bid_value than bid_b
        self.assertGreater(self.bid_a, self.bid_b)
        self.assertLess(self.bid_b, self.bid_a)
        
        # test tie-breaking by bidder name
        self.assertGreater(self.bid_b, self.bid_c) 
        self.assertLess(self.bid_c, self.bid_b) 

    def test_equality_same_winner_and_value(self):
        """__eq__ is based on winning_bidder and winning_bid."""
        other = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-C",
            bid_value=999.0,             # different internal bid_value
            winning_bidder="sat-A",
            winning_bid=10.0,            # same as bid_a
            t_img=200.0,
            t_update=5.0,
            performed=False,
        )
        self.assertEqual(self.bid_a, other)
        self.assertFalse(self.bid_a != other)

    def test_inequality_different_winner_or_value(self):
        """Bids should be unequal if winner or winning_bid differs."""
        self.assertNotEqual(self.bid_a, self.bid_b)
        self.assertTrue(self.bid_a != self.bid_b)

    # ---------- Modifiers / state updates ----------

    def test_set_updates_winner_and_times(self):
        """set() should update winning bid, winner, and timing fields."""
        self.bid_b.set(new_bid=42.0, t_img=150.0, t_update=3.5)

        self.assertEqual(self.bid_b.winning_bid, 42.0)
        self.assertEqual(self.bid_b.winning_bidder, "sat-B")  # bidder itself
        self.assertEqual(self.bid_b.t_img, 150.0)
        self.assertEqual(self.bid_b.t_update, 3.5)
        self.assertTrue(self.bid_b.has_winner())

    def test_set_performed_marks_performed_and_sets_winner(self):
        """set_performed() should flag performed and set winner/ time."""
        self.bid_a.set_performed(t=200.0, performed=True, performer="sat-Z")

        self.assertTrue(self.bid_a.performed)
        self.assertEqual(self.bid_a.winning_bidder, "sat-Z")
        self.assertEqual(self.bid_a.t_img, 200.0)

    def test_has_winner_flag(self):
        """has_winner() should check winning_bidder != NONE."""
        no_winner_bid = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-X",
        )
        self.assertFalse(no_winner_bid.has_winner())

        no_winner_bid.set(new_bid=5.0, t_img=50.0, t_update=1.0)
        self.assertTrue(no_winner_bid.has_winner())

    # ---------- update() and internal comparison logic ----------

    def test_update_time_only_when_other_is_worse(self):
        """
        If other has lower winning_bid but is a valid competing winner,
        UPDATE_TIME should be triggered (per your logic).
        """
        # self.bid_a currently: winning_bid = 10, bidder/winner = 'sat-A'
        worse_other = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-B",
            bid_value=7.0,
            winning_bidder="sat-B",
            winning_bid=7.0,
            t_img=120.0,
            t_update=5.0,
            performed=False,
        )

        # This relies on your __compare rules; mainly we expect not to lose our better winning bid.
        updated = self.bid_a.update(worse_other, t=10.0)

        # our winning info should remain, but time may be updated (depending on exact branch)
        self.assertEqual(updated.winning_bidder, "sat-A")
        self.assertEqual(updated.winning_bid, 10.0)
        self.assertGreaterEqual(updated.t_update, self.bid_a.t_update)

    def test_update_on_better_external_bid(self):
        """
        If other has a higher winning_bid, UPDATE should cause us to adopt it.
        """
        better_other = Bid(
            task=self.task,
            main_measurement="VNIR",
            bidder="sat-B",
            bid_value=20.0,
            winning_bidder="sat-B",
            winning_bid=20.0,
            t_img=130.0,
            t_update=5.0,
            performed=False,
        )

        updated = self.bid_a.update(better_other, t=10.0)

        self.assertEqual(updated.winning_bidder, "sat-B")
        self.assertEqual(updated.winning_bid, 20.0)
        self.assertEqual(updated.t_img, 130.0)
        self.assertEqual(updated.t_update, 10.0)

    # ---------- String representations ----------

    def test_str_contains_key_fields(self):
        """__str__ should include task id, bidder, winner, and times."""
        s = str(self.bid_a)
        self.assertIn("task_id", s)
        self.assertIn("bidder", s)
        self.assertIn("winner", s)
        self.assertIn("t_img", s)
        self.assertIn("t_update", s)

    def test_repr_is_hashable_and_includes_id(self):
        """__repr__ should be stable and usable in hashing."""
        r = repr(self.bid_a)
        h = hash(self.bid_a)  # shouldn't raise
        self.assertIsInstance(h, int)
        self.assertIn("Bid_", r)
        self.assertIn("sat-A", r)

    # ---------- Compare Method ----------

    def test_compare_when_other_performed_updates(self):
        """If other is marked performed and self is not, compare() should request UPDATE."""
        base = self.bid_a.copy()
        other = self.bid_a.copy()
        self.assertFalse(base.performed)
        other.performed = True

        comp, reb = base.compare(other)

        self.assertEqual(comp, Bid.UPDATE)
        self.assertEqual(reb, Bid.REBROADCAST_OTHER)

    def test_compare_same_bidder_newer_timestamp(self):
        """Same bidder + newer t_update => UPDATE + REBROADCAST_OTHER."""
        a_newer = self.bid_a.copy()
        a_older = self.bid_a.copy()
        a_newer.t_update = 10.0
        a_older.t_update = 5.0

        comp, reb = a_older.compare(a_newer)

        self.assertEqual(comp, Bid.UPDATE)
        self.assertEqual(reb, Bid.REBROADCAST_OTHER)

    def test_compare_same_bidder_older_timestamp(self):
        """Same bidder + older t_update => LEAVE + NO_REBROADCAST."""
        a_newer = self.bid_a.copy()
        a_older = self.bid_a.copy()
        a_newer.t_update = 5.0
        a_older.t_update = 10.0

        comp, reb = a_older.compare(a_newer)

        self.assertEqual(comp, Bid.LEAVE)
        self.assertEqual(reb, Bid.NO_REBROADCAST)


if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Bid Test')
    
    # run tests
    unittest.main()
