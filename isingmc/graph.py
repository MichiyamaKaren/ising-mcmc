from typing import List, MutableSet, Optional


class Vertex:
    def __init__(self, edges: Optional[MutableSet['Edge']] = None) -> None:
        if edges is None:
            self.edges = set()
        else:
            self.edges = edges

    def add_edge(self, edge: 'Edge'):
        self.edges.add(edge)
    
    def search_adjcent_edge(self, vex: 'Vertex') -> 'Edge':
        for edge in self.edges:
            if edge.same_edge(self, vex):
                return edge
        return None


class Edge:
    def __init__(self, headvex: Vertex, tailvex: Vertex) -> None:
        self.headvex = headvex
        self.tailvex = tailvex
    
    def same_edge(self, headvex: Vertex, tailvex: Vertex) -> bool:
        return ((headvex is self.headvex) and (tailvex is self.tailvex)) or \
            ((headvex is self.tailvex) and (tailvex is self.headvex))


class Graph:
    def __init__(self, vertexes: List[Vertex],
                 edges: Optional[List[Edge]] = None) -> None:
        self.vertexes = vertexes
        if edges is None:
            self.edges = []
        else:
            self.edges = edges

    def _append_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        edge.headvex.add_edge(edge)
        edge.tailvex.add_edge(edge)

    def _create_edge(self, headvex: Vertex, tailvex: Vertex) -> Edge:
        # likely a 'edge factory'
        return Edge(headvex, tailvex)

    def add_edge(self, headvex_i: int, tailvex_i: int) -> None:
        headvex = self.vertexes[headvex_i]
        tailvex = self.vertexes[tailvex_i]

        if headvex.search_adjcent_edge(tailvex) is not None:
            # edge already exists
            return

        new_edge = self._create_edge(headvex, tailvex)
        self._append_edge(new_edge)