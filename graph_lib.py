import collections
import heapq
import random

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
        for w in graph.vertices():
            if v == w: continue
            for j in shortest_path(graph, v, w):
                centrality[j] += 1
    return centrality


def dijkstra(graph, A, B):
    heap = []
    return None


def _reconstruct_path(parents, B):
    stack = []
    curr = B
    while curr:
        stack.append(curr)
        curr = parents[curr]
    return stack[::-1]


def shortest_path_bfs(graph, A, B):
    parents = {A: None}
    queue = collections.deque([A])
    while queue:
        v = queue.pop()
        for w in graph.adjacent(v):
            if w in parents: continue
            parents[w] = v
            queue.appendleft(w)
            if w == B:
                return _reconstruct_path(parents, B)
    return []


def shortest_path(graph, A, B):
    if graph.is_weighted():
        return dijkstra(graph, A, B)
    return shortest_path_bfs(graph, A, B)


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


def export_kml(graph, path):
    return