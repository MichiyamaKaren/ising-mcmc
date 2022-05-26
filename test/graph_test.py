import unittest

from isingmc import Graph, Vertex


class GraphTest(unittest.TestCase):
    def test_add_edge(self):
        vexes = [Vertex() for _ in range(3)]
        g = Graph(vexes)
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        g.add_edge(1, 0)

        self.assertEqual(len(g.edges), 2)
        self.assertEqual(len(vexes[0].edges), 2)
        self.assertEqual(len(vexes[1].edges), 1)
        self.assertEqual(len(vexes[2].edges), 1)
