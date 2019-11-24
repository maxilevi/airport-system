import collections
import heapq
import random
from graph import Graph

RANDOM_WALK_ITERATIONS = 20

def page_rank(graph):
    return

def _build_centrality_dict(graph):
    centrality = {}
    for v in graph.vertices():
        centrality[v] = 0
    return centrality


def approximate_centrality(graph):
    centrality = _build_centrality_dict(graph)
    for v in graph.vertices():
        for j in random_walk(graph, v, RANDOM_WALK_ITERATIONS):
            centrality[j] += 1
    return centrality


def betweeness_centrality(graph):
    centrality = _build_centrality_dict(graph)
    for v in graph.vertices():
        dists, parents = dijkstra(graph, v)
        aux = _build_centrality_dict(graph)

        sorted_vertices = sorted(graph.vertices(), key=lambda x: dists[x])
        for w in sorted_vertices:
            if w in parents and parents[w]:
                aux[parents[w]] += 1 + aux[w]

        for w in graph.vertices():
            if w == v: continue
            centrality[w] += aux[w]

    return centrality


def dijkstra(graph, A):
    parents = {}
    dists = {}
    for v in graph.vertices():
        dists[v] = float('inf')
    dists[A] = 0
    parents[A] = None
    heap = []
    heapq.heappush(heap, (dists[A], A))
    while heap:
        _, v = heapq.heappop(heap)
        for w in graph.adjacent(v):
            new_dist = dists[v] + graph.weight((v, w))
            if new_dist < dists[w]:
                dists[w] = new_dist
                parents[w] = v
                heapq.heappush(heap, (-dists[w], w))
    return dists, parents


def _reconstruct_path(parents, B):
    stack = []
    curr = B
    while curr:
        stack.append(curr)
        curr = parents[curr]
    return stack[::-1]


def shortest_path_bfs(graph, A):
    parents = {A: None}
    queue = collections.deque([A])
    while queue:
        v = queue.pop()
        for w in graph.adjacent(v):
            if w in parents: continue
            parents[w] = v
            queue.appendleft(w)
    return parents


def shortest_path(graph, A, B):
    if graph.is_weighted():
        dists, parents = dijkstra(graph, A)
    else:
        parents = shortest_path_bfs(graph, A)
    return _reconstruct_path(parents, B)


def random_walk(graph, A, iterations):
    if graph.is_weighted():
        raise NotImplementedError('')

    parent = None
    parents = {A: parent}
    curr = A
    for i in range(iterations):
        neighbours = graph.adjacent(curr)
        if neighbours:
            parent, curr = curr, neighbours[random.randint(0, len(graph.adjacent(A))-1)]
            parents[curr] = parent
    return _reconstruct_path(parents, curr)


# Me gusta mas kruskal, pero ni ganas de programarme un disjoint set
def _prim(graph):
    # Creamos el arbol y le añadimos un vertice cualquiera.
    mst = Graph(is_undirected=True)
    heap = []
    random_vertex = graph.random_vertex()
    mst.add_vertex(random_vertex)

    def queue_neighbours(v):
        for w in graph.adjacent(v):
            heapq.heappush(heap, (graph.weight((v, w)), (v, w)))

    queue_neighbours(random_vertex)
    visited = set([random_vertex])

    while heap:
        weight, edge = heapq.heappop(heap)
        v, w = edge
        if w not in visited:
            visited.add(w)
            queue_neighbours(w)
            mst.add_vertex(w)
            mst.add_edge((v, w), weight=weight)
    return mst


def build_MST(graph):
    if not graph.is_undirected():
        raise ValueError('Los digrafos no tienen MSTs')
    return _prim(graph)

def export_kml(graph, path):
    return