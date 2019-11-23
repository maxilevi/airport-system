class Graph:

    def __init__(self):
        self._adjacency_list = {}
        self._is_weighted = False

    def vertices(self):
        return list(self._adjacency_list.keys())

    def adjacent(self, vertex):
        return list([x for x in self._adjacency_list[vertex].keys()])

    def add_vertex(self, vertex):
        if vertex not in self._adjacency_list:
            self._adjacency_list[vertex] = {}

    def add_edge(self, edge, weight=1):
        x, y = edge
        self._adjacency_list[x][y] = weight
        self._is_weighted = self._is_weighted or weight != 1

    def weight(self, edge):
        x, y = edge
        return self._adjacency_list[x][y]

    def random_vertex(self):
        return None

    def is_weighted(self):
        return self._is_weighted