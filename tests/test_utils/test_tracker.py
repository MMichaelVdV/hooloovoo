import unittest

from hooloovoo.utils.track import Tracker


class TestTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = Tracker()
        self.has_grown = self.tracker.track(grown=0)(lambda old, new: new > old)
        self.has_shrunk = self.tracker.track(shrunk=0)(lambda old, new: new < old, auto_update=False)
        self.has_changed = self.tracker.track(changed=False)(lambda old, new: new != old)

        self.longer_ref = ["very_long"]
        self.longer = self.tracker.track(longer="short")(condition=lambda old, new: len(new) > len(old),
                                                         new=lambda: self.longer_ref[0])
        self.shorter_ref = ["short"]
        self.shorter = self.tracker.track(shorter="long text")(condition=lambda old, new: len(new) < len(old),
                                                               new=lambda: self.shorter_ref[0])

    def test_basics(self):
        self.assertFalse(self.has_grown(0))
        self.assertFalse(self.has_grown(-1))
        self.assertEqual(-1, self.has_grown.value)
        self.assertTrue(self.has_grown(1))
        self.assertEqual(1, self.has_grown.value)
        self.assertFalse(self.has_grown(1))
        self.assertEqual(1, self.has_grown.value)
        self.has_grown.update(10)
        self.assertEqual(10, self.has_grown.value)
        self.assertFalse(self.has_grown(10))
        self.assertTrue(self.has_grown(11))
        self.assertEqual(11, self.has_grown.value)

        self.assertFalse(self.has_shrunk(0))
        self.assertTrue(self.has_shrunk(-1))
        self.assertEqual(0, self.has_shrunk.value)  # still the same, auto_update is off
        self.assertTrue(self.has_shrunk(-1))
        self.has_shrunk.update(-10)
        self.assertEqual(-10, self.has_shrunk.value)
        self.assertFalse(self.has_shrunk(-9))
        self.assertTrue(self.has_shrunk(-11))
        self.assertEqual(-10, self.has_shrunk.value)

    def test_new_cb_1(self):
        self.assertTrue(self.longer())
        self.assertFalse(self.longer())

    def test_new_cb_2(self):
        self.longer.update()
        self.assertFalse(self.longer())
        self.longer_ref = ["extremely_long"]
        self.assertTrue(bool(self.longer))
        self.assertFalse(bool(self.longer))

    def test_new_cb_3(self):
        with self.assertRaises(RuntimeError):
            self.has_grown.update()
        with self.assertRaises(RuntimeError):
            self.has_grown()
        with self.assertRaises(RuntimeError):
            bool(self.has_grown)

    def test_exp_1(self):
        exp = (self.has_grown | self.has_shrunk) & -self.has_changed
        # print(exp)

        b1 = exp.eval(False)(grown=1, shrunk=-1, changed=False)
        self.assertTrue(b1)  # condition satisfied, none of the values update (due to False in eval)
        self.assertEqual({'grown': 0, 'shrunk': 0, 'changed': False}, exp.values())

        b2 = exp.eval(True)(grown=1, shrunk=-1, changed=False)
        self.assertTrue(b2)  # condition satisfied, all values update (due to True in eval)
        self.assertEqual({'grown': 1, 'shrunk': -1, 'changed': False}, exp.values())

        b3 = exp.eval()(grown=2, shrunk=-2, changed=False)
        self.assertTrue(b3)  # condition satisfied, values with auto_update enable change (due to None in eval)
        self.assertEqual({'grown': 2, 'shrunk': -1, 'changed': False}, exp.values())

        b4 = exp.eval(True)(grown=3, shrunk=-3, changed=True)
        self.assertFalse(b4)  # condition not satisfied, all values update (due to True in eval)
        self.assertEqual({'grown': 3, 'shrunk': -3, 'changed': True}, exp.values())

    def test_exp_2(self):
        exp = self.longer & self.has_grown
        # print(exp)

        b1 = exp.eval(True)(grown=-1)
        self.assertFalse(b1)
        self.assertEqual({'longer': "very_long", 'grown': -1}, exp.values())

        self.longer.update("short")
        self.assertTrue(bool(exp(grown=1)))
        self.assertEqual({'longer': "very_long", 'grown': 1}, exp.values())

        exp.update_all(longer="short", grown=0)
        self.assertEqual({'longer': "short", 'grown': 0}, exp.values())

        exp.update_all(grown=1)
        self.assertEqual({'longer': "very_long", 'grown': 1}, exp.values())

    def test_exp_3(self):
        exp = self.longer & self.shorter
        if exp:
            self.assertTrue(True)
        else:
            raise AssertionError("exp shoud have been True")

        # the values were updated, so it won't trigger again
        self.assertFalse(exp)

    def test_exp_4(self):
        exp = self.longer & self.shorter
        exp.update_all()
        self.assertFalse(exp)

    # noinspection PyStatementEffect
    def test_example(self):
        # imagine this is constantly updated by e.g. a sensor
        apple_tree = {
            "size": 1.4,
            "n_apples": 0,
            "healthy": True
        }

        status = Tracker()
        has_grown = status.track(grown=0)(lambda o, n: n > o, new=lambda: apple_tree["size"])
        more_apples = status.track(apples=0)(lambda o, n: n > o, new=lambda: apple_tree["n_apples"])
        is_healthy = status.track(healthy=True)(lambda _, n: n, new=lambda: apple_tree["healthy"])

        good = (has_grown | more_apples) & is_healthy
        good  # default values

        self.assertTrue(good)  # it is good because the tree has grown
        good

        self.assertFalse(good)  # the tree did not grow since the last check, not good

        apple_tree["size"] = 2.8
        apple_tree["n_apples"] = 10
        self.assertTrue(good)  # bigger tree and more apples, good
        good

        apple_tree["size"] = 3.2
        apple_tree["healthy"] = False
        self.assertFalse(good)  # the tree has grown but is sick, not good
        good
