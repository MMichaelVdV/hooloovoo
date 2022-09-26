import pickle
import unittest

from hooloovoo.utils.tree import Tree, t


class TestTree(unittest.TestCase):

    def test_basics(self):
        tr = Tree()

        # reports missing keys
        with self.assertRaises(KeyError): tr.foo
        with self.assertRaises(KeyError): tr["foo"]

        # supports both dict and property syntax
        tr["foo"] = 1
        tr.bar = 2
        self.assertTrue(tr.foo == tr["foo"] == 1)
        self.assertTrue(tr.bar == tr["bar"] == 2)

        # supports deletion
        del tr["foo"]
        del tr.bar
        with self.assertRaises(KeyError): tr.foo
        with self.assertRaises(KeyError): tr["foo"]

    def test_default(self):
        tr = Tree()

        # if no value for the key exists, then set the default value
        answer1 = tr.with_default(42).answer
        self.assertTrue(answer1 == tr.answer == 42)

        # a value for the key exists, keep old value
        answer2 = tr.with_default(666).answer
        self.assertTrue(answer1 == answer2 == tr.answer == 42)

        # dict syntax works
        foo1 = tr.with_default("foo")["foo"]
        self.assertTrue(foo1 == tr.foo == "foo")
        foo2 = tr.with_default("bar")["foo"]
        self.assertTrue(foo1 == foo2 == tr.foo == "foo")

    def test_default_assign(self):
        tr = Tree()
        # doing this makes no sense,
        # because why would you need a default value if you assign it with something else anyway?
        # It does makes sense for allowing updating a default value. (+=)
        tr.with_default(0).nonsense = 1
        self.assertEqual(tr.nonsense, 1)

    def test_default_update(self):
        tr = Tree()
        tr.with_default(0).counter += 1
        tr.with_default(0).counter += 1
        self.assertEqual(tr.counter, 2)

    def test_complex_keys(self):
        tr = Tree()
        k = "!@#$%^&*()-=+ definitely not a valid python variable name"

        tr[k] = 42
        self.assertEqual(tr[k], 42)
        del tr[k]
        v = tr.with_default(10)[k]
        self.assertTrue(v == tr[k] == 10)

    def test_constructor(self):
        tr = Tree({
            "foo": 1,
            "bar": 2,
        })
        self.assertEqual(tr.foo, 1)
        self.assertEqual(tr.bar, 2)

    def test_nested(self):
        tr = Tree({
            "foo": 1,
            "bar": {
                "baz": 2,
                "bux": 3
            }
        })
        self.assertEqual(tr.foo, 1)
        self.assertEqual(tr.bar.baz, 2)
        self.assertEqual(tr.bar.bux, 3)

    def test_nested_list(self):
        tr = Tree({
            "foo": 1,
            "bar": [
                {"x": 1, "y": 2},
                {"x": 1, "y": 2},
                {"x": 1, "y": [1, 2]},
            ]
        })
        # print(tr)

    def test_to_dict(self):
        d = {
            "foo": 1,
            "bar": {
                "baz": 2,
                "bux": 3
            }
        }
        tr = Tree(d)
        self.assertEqual(tr.to_dict(), d)

    def test_pickle(self):
        d = {
            "foo": 1,
            "bar": {
                "baz": 2,
                "bux": 3
            }
        }
        t0 = Tree(d)
        t1 = pickle.loads(pickle.dumps(t0))
        self.assertEqual(d, t1.to_dict())
        self.assertIsInstance(t1.bar, Tree)

    def test_t(self):
        tr = t(x=1, y=2)
        self.assertEqual(Tree({"x": 1, "y": 2}), tr)

    def test_unpack(self):
        def foo(**kwargs):
            return kwargs

        d = dict(x=1, y=2)
        tr = Tree(d)
        # noinspection PyArgumentList
        self.assertEqual(foo(**tr), d)

    def test_deepcopy(self):
        tr = Tree({
            "foo": 1,
            "bar": [1, 2],
            "baz": {
                "a": [3, 4],
                "b": [
                    {"x": 1, "y": 2}
                ]
            }
        })
        tr2 = tr.deepcopy()
        tr2.foo = 10
        self.assertEqual(1, tr.foo)

        tr2.bar[0] = 10
        self.assertEqual(1, tr.bar[0])

        tr2.baz.a[0] = 10
        self.assertEqual(3, tr.baz.a[0])

        tr2.baz.b[0].x = 10
        self.assertEqual(1, tr.baz.b[0].x)


if __name__ == '__main__':
    unittest.main()
