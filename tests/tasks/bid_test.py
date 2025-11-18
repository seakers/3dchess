import unittest


# Adjust this import path to wherever your Bid class lives
from chess3d.utils import print_welcome
from chess3d.agents.planning.decentralized.consensus.bids import Bid
from chess3d.agents.planning.tasks import DefaultMissionTask


class TestBids(unittest.TestCase):
    """
    Test suite for bids used in consensus planners.
    """

    def setUp(self):
        # Common task used across many tests
        self.task_a = DefaultMissionTask(parameter='test_param',
                                       location=(0.0, 0.0, 0, 0),
                                       mission_duration=100.0,
                                       priority=1)
        
        self.task_b = DefaultMissionTask(parameter='test_param',
                                       location=(0.0, 0.0, 1, 2),
                                       mission_duration=50.0,
                                       priority=0.5)
        
    # ----------------------------------------------------------------------
    # Helper to build bids quickly
    # ----------------------------------------------------------------------
    def make_bid(self,
                 bidder="agentA",
                 winning_bidder=Bid.NONE,
                 winning_bid=0.0,
                 bid_value=None,
                 t_img=0.0,
                 n_img=0,
                 t_stamp=0.0,
                 performed=False,
                 main_measurement="VNIR",
                 task=None,
                ) -> Bid:
        task = task or self.task_a
        if bid_value is None:
            bid_value = winning_bid

        return Bid(
            task=task,
            main_measurement=main_measurement,
            bidder=bidder,
            bid_value=bid_value,
            winning_bidder=winning_bidder,
            winning_bid=winning_bid,
            t_img=t_img,
            n_img=n_img,
            t_stamp=t_stamp,
            performed=performed,
        )

    # ----------------------------------------------------------------------
    # Basic construction / serialization
    # ----------------------------------------------------------------------
    def test_to_dict_and_from_dict_roundtrip(self):
        bid = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            t_img=10.0,
            n_img=1,
            t_stamp=5.0,
            performed=True,
        )

        bid_dict = bid.to_dict()
        reconstructed = Bid.from_dict(bid_dict)

        self.assertEqual(bid.task.id, reconstructed.task.id)
        self.assertEqual(bid.main_measurement, reconstructed.main_measurement)
        self.assertEqual(bid.bidder, reconstructed.bidder)
        self.assertAlmostEqual(bid.bid_value, reconstructed.bid_value)
        self.assertEqual(bid.winning_bidder, reconstructed.winning_bidder)
        self.assertAlmostEqual(bid.winning_bid, reconstructed.winning_bid)
        self.assertAlmostEqual(bid.t_img, reconstructed.t_img)
        self.assertEqual(bid.n_img, reconstructed.n_img)
        self.assertAlmostEqual(bid.t_stamp, reconstructed.t_stamp)
        self.assertEqual(bid.performed, reconstructed.performed)

    def test_copy_creates_independent_object(self):
        bid = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            t_img=10.0,
            n_img=1,
            t_stamp=5.0,
            performed=False,
        )
        bid_copy = bid.copy()

        self.assertIsNot(bid, bid_copy)
        self.assertEqual(bid, bid_copy)
        # mutate copy to ensure original is unchanged
        bid_copy.set(20.0, t_img=15.0, t_update=6.0)
        self.assertNotEqual(bid.winning_bid, bid_copy.winning_bid)

    # ----------------------------------------------------------------------
    # Comparison operators (value + tie-breaker)
    # ----------------------------------------------------------------------
    def test_comparison_operators_basic(self):
        # b_high has higher bid_value than b_low
        b_low = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=5.0,
            bid_value=5.0,
            t_stamp=1.0,
        )
        b_high = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            bid_value=10.0,
            t_stamp=1.0,
        )

        self.assertTrue(b_high > b_low)
        self.assertTrue(b_low < b_high)
        self.assertTrue(b_high >= b_low)
        self.assertTrue(b_low <= b_high)
        self.assertFalse(b_high == b_low)
        self.assertTrue(b_high != b_low)

    def test_comparison_tie_breaker_on_bidder_name(self):
        # Same winning_bid and bid_value; tie broken on bidder name (alphabetical)
        bA = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            bid_value=10.0,
            t_stamp=1.0,
        )
        bB = self.make_bid(
            bidder="agentB",
            winning_bidder="agentB",
            winning_bid=10.0,
            bid_value=10.0,
            t_stamp=1.0,
        )

        # __tie_breaker returns the lexicographically smallest bidder as "greater"
        # because it uses min(...) and __gt__/__lt__ interpret it accordingly.
        self.assertTrue(bA < bB)
        self.assertTrue(bB > bA)

    def test_comparison_input_checks(self):
        b1 = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            bid_value=10.0,
            t_stamp=1.0,
            task=self.task_a,
        )
        b2 = self.make_bid(
            bidder="agentB",
            winning_bidder="agentB",
            winning_bid=5.0,
            bid_value=5.0,
            t_stamp=1.0,
            task=self.task_b,  # different task
        )
        b3 = self.make_bid(
            bidder="agentA",
            winning_bidder="agentA",
            winning_bid=10.0,
            bid_value=10.0,
            t_stamp=1.0,
            task=self.task_a,
            n_img=10 # different n_img
        )

        # compare to b2 (different task) should raise AssertionError
        with self.assertRaises(AssertionError): b1 > "b2"
        with self.assertRaises(AssertionError): b1 > b2
        with self.assertRaises(AssertionError): b1 < b2
        with self.assertRaises(AssertionError): b1 >= b2
        with self.assertRaises(AssertionError): b1 <= b2

        # compare to b3 (different n_img) should raise AssertionError
        with self.assertRaises(AssertionError): "b1" > b3
        with self.assertRaises(AssertionError): b1 > b3
        with self.assertRaises(AssertionError): b1 < b3
        with self.assertRaises(AssertionError): b1 >= b3
        with self.assertRaises(AssertionError): b1 <= b3

    # ----------------------------------------------------------------------
    # compare() behavior in representative scenarios
    # ----------------------------------------------------------------------
    def test_compare_sender_thinks_itself_winner(self):
        """
        CASE 1: Sender thinks itself winner
        """
        bid_sender = self.make_bid(
            bidder="sendingAgent",
            winning_bidder="sendingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )
        # CASE 1.1: Receiver thinks itself winner 
        # CASE 1.1.1: Equal bids
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   
        
        # CASE 1.1.1.1: Receiver's bidder name is lexicographically smaller
        # Expect: LEAVE, NO_REBROADCAST
        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.1.1.2: Sender's bidder name is lexicographically smaller
        # Expect: UPDATE, REBROADCAST_OTHER
        action, rebcast = bid_sender.compare(bid_receiver)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.1.2: Sender has higher bid
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=1.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.1.3: Receiver has higher bid
        # Expect: UPDATE_TIME, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=100.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE_TIME)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.2: Receiver thinks sender is the winner 
        # CASE 1.2.1: Equal time-stamps
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="sendingAgent",
            winning_bid=100.0,
            t_stamp=2.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.2.2: Receiver has older time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="sendingAgent",
            winning_bid=100.0,
            t_stamp=1.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.2.3: Receiver has newer time-stamp
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="sendingAgent",
            winning_bid=100.0,
            t_stamp=3.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.3: Receiver thinks a third party is the winner 
        # CASE 1.3.1: Equal bids
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.3.2: Sender has higher bid
        # CASE 1.3.2.1: Sender has newer or equal time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=1.0,
            t_stamp=2.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=1.0,
            t_stamp=1.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3.2.2: Receiver has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=1.0,
            t_stamp=3.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3.3: Receiver has higher bid
        # CASE 1.3.3.1: Receiver has newer or equal time-stamp
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=100.0,
            t_stamp=2.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=100.0,
            t_stamp=3.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.3.3.2: Sender has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=100.0,
            t_stamp=1.0,
        )

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.4: Receiver has no winner
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder=Bid.NONE,
            winning_bid=0.0,
            t_stamp=1.0,
        )       
        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

    def test_compare_sender_thinks_receiver_winner(self):
        """
        CASE 1: Sender thinks the receiving agent is the winner
        """
        bid_sender = self.make_bid(
            bidder="sendingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )
        # CASE 1.1: Receiver thinks itself winner 
        # CASE 1.1.1: Equal time-stamps
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.1.2: Sender has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=1.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.1.3: Receiver has newer time-stamp
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=3.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.2: Receiver thinks sender is the winner 
        # Expect: RESET, REBROADCAST_EMPTY
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="sendingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.RESET)
        self.assertEqual(rebcast, Bid.REBROADCAST_EMPTY)

        # CASE 1.3: Receiver thinks a third party is the winner 
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.4: Receiver has no winner
        # Expect: LEAVE, REBROADCAST_EMPTY
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder=Bid.NONE,
            winning_bid=0.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_EMPTY)

    def test_compare_sender_thinks_third_party_winner(self):
        bid_sender = self.make_bid(
            bidder="a_sendingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )

        # CASE 1.1: Receiver thinks itself winner
        # CASE 1.1.1: Equal Bids
        # CASE 1.1.1.1: Sender's bidder name is lexicographically smaller
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.1.1.2: Receiver's bidder name is lexicographically smaller
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="a_a_receivingAgent",
            winning_bidder="a_a_receivingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.1.2: Sender has higher bid
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=1.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.1.3: Receiver has higher bid
        # Expect: UPDATE_TIME, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=100.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE_TIME)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.2: Receiver thinks sender is the winner
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="a_sendingAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3: Receiver thinks a third party is the winner 
        # CASE 1.3.1: Equal time-stamps
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.3.2: Sender has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=1.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3.3: Receiver has newer time-stamp
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=10.0,
            t_stamp=3.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.4: Receiver thinks a fourth party is the winner 
        # CASE 1.4.1: Equal bids
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=10.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)

        # CASE 1.4.2: Sender has higher bid
        # CASE 1.4.2.1: Sender has equal or older time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=1.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=1.0,
            t_stamp=1.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.4.2.2: Receiver has newer time-stamp
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=1.0,
            t_stamp=3.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.4.3: Receiver has higher bid
        # CASE 1.4.3.1: Receiver has equal or newer time-stamp
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=100.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=100.0,
            t_stamp=3.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.4.3.1: Sender has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="fourthPartyAgent",
            winning_bid=100.0,
            t_stamp=1.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.5: Receiver has no winner
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder=Bid.NONE,
            winning_bid=0.0,
            t_stamp=2.0,
        )   

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

    def test_compare_sender_has_no_winner(self):
        bid_sender = self.make_bid(
            bidder="sendingAgent",
            winning_bidder=Bid.NONE,
            winning_bid=0.0,
            t_stamp=2.0,
        )

        # CASE 1.1: Receiver thinks itself winner
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="receivingAgent",
            winning_bid=0.0,
            t_stamp=2.0,
        )  

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.2: Receiver thinks sender is the winner
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="sendingAgent",
            winning_bid=0.0,
            t_stamp=2.0,
        )  

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3: Receiver thinks a third party is the winner 
        # CASE 1.3.1: Sender has newer time-stamp
        # Expect: UPDATE, REBROADCAST_OTHER
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=0.0,
            t_stamp=1.0,
        )  

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.UPDATE)
        self.assertEqual(rebcast, Bid.REBROADCAST_OTHER)

        # CASE 1.3.2: Otherwise
        # Expect: LEAVE, REBROADCAST_SELF
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder="thirdPartyAgent",
            winning_bid=0.0,
            t_stamp=3.0,
        )  

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.REBROADCAST_SELF)

        # CASE 1.4: Receiver has no winner
        # Expect: LEAVE, NO_REBROADCAST
        bid_receiver = self.make_bid(
            bidder="receivingAgent",
            winning_bidder=Bid.NONE,
            winning_bid=0.0,
            t_stamp=2.0,
        )  

        action, rebcast = bid_receiver.compare(bid_sender)
        self.assertEqual(action, Bid.LEAVE)
        self.assertEqual(rebcast, Bid.NO_REBROADCAST)


    # # ----------------------------------------------------------------------
    # # Modifiers: update / set / set_performed / has_winner
    # # ----------------------------------------------------------------------
    # def test_update_applies_update_info(self):
    #     """
    #     Scenario where compare() returns UPDATE and we apply update().
    #     """
    #     base_bid = self.make_bid(
    #         bidder="agentA",
    #         winning_bidder="agentA",
    #         winning_bid=5.0,
    #         t_img=10.0,
    #         t_stamp=1.0,
    #     )
    #     other_bid = self.make_bid(
    #         bidder="agentB",
    #         winning_bidder="agentB",
    #         winning_bid=10.0,
    #         t_img=20.0,
    #         t_stamp=2.0,
    #     )

    #     # sanity check: compare says UPDATE, REBROADCAST_OTHER
    #     action, rebcast = base_bid.compare(other_bid)
    #     self.assertEqual(action, Bid.UPDATE)

    #     updated = base_bid.update(other_bid, t=3.0)

    #     self.assertAlmostEqual(updated.winning_bid, other_bid.winning_bid)
    #     self.assertEqual(updated.winning_bidder, other_bid.winning_bidder)
    #     self.assertAlmostEqual(updated.t_img, other_bid.t_img)
    #     self.assertAlmostEqual(updated.t_stamp, 3.0)

    # def test_set_and_has_winner(self):
    #     bid = self.make_bid(
    #         bidder="agentA",
    #         winning_bidder=Bid.NONE,
    #         winning_bid=0.0,
    #         t_img=np.NINF,
    #         t_stamp=0.0,
    #     )

    #     self.assertFalse(bid.has_winner())

    #     bid.set(new_bid=7.5, t_img=12.0, t_update=3.0)

    #     self.assertTrue(bid.has_winner())
    #     self.assertEqual(bid.winning_bidder, "agentA")
    #     self.assertAlmostEqual(bid.winning_bid, 7.5)
    #     self.assertAlmostEqual(bid.t_img, 12.0)
    #     self.assertAlmostEqual(bid.t_stamp, 3.0)

    # def test_set_performed_marks_performed_and_updates_time(self):
    #     bid = self.make_bid(
    #         bidder="agentA",
    #         winning_bidder=Bid.NONE,
    #         winning_bid=0.0,
    #         t_img=np.NINF,
    #         t_stamp=0.0,
    #     )

    #     bid.set_performed(t=50.0, performed=True, performer="agentB")

    #     self.assertTrue(bid.performed)
    #     self.assertEqual(bid.winning_bidder, "agentB")
    #     self.assertAlmostEqual(bid.t_img, 50.0)
    #     self.assertAlmostEqual(bid.t_stamp, 50.0)


if __name__ == '__main__':
    # terminal welcome message
    print_welcome('Task Bid Test')
    
    # run tests
    unittest.main()
