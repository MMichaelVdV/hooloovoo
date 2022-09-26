import pickle
import unittest

from hooloovoo.utils.timer import Timer, TimerState


class TestTimer(unittest.TestCase):

    def test_basics(self):
        t = Timer()
        self.assertEqual(t.state, TimerState.INACTIVE)
        self.assertEqual(t.duration, 0)
        t.start()
        self.assertEqual(t.state, TimerState.ACTIVE)
        self.assertGreater(t.duration, 0)
        cp0 = t.peek
        cp1 = t.pause()
        self.assertEqual(t.state, TimerState.PAUSED)
        self.assertEqual(cp1.duration, t.duration)
        self.assertGreater(cp1.duration, cp0.duration)

    def test_pickle(self):
        t0 = Timer()
        t1 = pickle.loads(pickle.dumps(t0))
        self.assertEqual(t0.state, t1.state)
        self.assertEqual(t1.duration, 0)

        t0.start()
        t2 = pickle.loads(pickle.dumps(t0))
        self.assertTrue(t0.state == t2.state == TimerState.ACTIVE)
        self.assertGreater(t0.duration, t2.duration)  # t2 was not running when being pickled, t0 kept going.

        t0.pause()
        t3 = pickle.loads(pickle.dumps(t0))
        self.assertTrue(t0.state == t3.state == TimerState.PAUSED)
        self.assertEqual(t0.duration, t3.duration)

        t0.lap()
        self.assertNotEqual(t0.checkpoints, t3.checkpoints)
        t4 = pickle.loads(pickle.dumps(t0))
        self.assertEqual(t0.checkpoints, t4.checkpoints)


if __name__ == '__main__':
    unittest.main()
