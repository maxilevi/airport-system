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
    heapq.heappush(heap, (-dists[A], A))
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
        if curr in parents:
            curr = parents[curr]
        else:
            # A & B are not connected
            return None
    return stack[::-1]


def shortest_path_bfs(graph, A):
    parents = {A: None}
    dists = {A: 0}
    queue = collections.deque([A])
    while queue:
        v = queue.pop()
        for w in graph.adjacent(v):
            if w in parents: continue
            parents[w] = v
            dists[w] = dists[v] + 1
            queue.appendleft(w)
    return dists, parents


def shortest_path(graph, A, B):
    if graph.is_weighted():
        dists, parents = dijkstra(graph, A)
    else:
        dists, parents = shortest_path_bfs(graph, A)
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


def _queue_neighbours(graph, heap, v):
    for w in graph.adjacent(v):
        heapq.heappush(heap, (graph.weight((v, w)), (v, w)))


# Me gusta mas kruskal, pero ni ganas de programarme un disjoint set
def _prim(graph):
    # Creamos el arbol y le añadimos un vertice cualquiera.
    mst = Graph(is_undirected=True)
    heap = []
    random_vertex = graph.random_vertex()
    mst.add_vertex(random_vertex)

    _queue_neighbours(graph, heap, random_vertex)
    visited = set([random_vertex])

    while heap:
        weight, edge = heapq.heappop(heap)
        v, w = edge
        if w not in visited:
            visited.add(w)
            _queue_neighbours(graph, heap, w)
            mst.add_vertex(w)
            mst.add_edge((v, w), weight=weight)
    return mst


def build_MST(graph):
    if not graph.is_undirected():
        raise ValueError('Los digrafos no tienen MSTs')
    return _prim(graph)


def path_visiting_every_vertex(graph, A):

    def visit(path, remaining, cost, min_cost, min_path):

        if len(remaining) == 0:
            return cost, path

        # PODA, si el costo es mayor o igual al minimo y todavia falta agregar, nunca va a poder llegar a ser mejor
        if cost >= min_cost:
            return None, None

        for i in range(len(remaining)):
            w = remaining[i]
            dists, parents = dijkstra(graph, path[-1])
            shortest = _reconstruct_path(parents, w)[1:]
            remaining_copy = remaining[:i] + remaining[i+1:]

            # Sacamos los vertices por los que el camino paso. Ya los visitamos
            for vertex in shortest:
                if vertex in remaining_copy:
                    remaining_copy.remove(vertex)

            curr_cost, curr_path = visit(path + shortest, remaining_copy, cost + dists[w], min_cost, min_path)
            if curr_path:
                if curr_cost < min_cost:
                    min_cost = curr_cost
                    min_path = curr_path

        return min_cost, min_path

    best_cost, best_path = visit([A], [x for x in graph.vertices() if x != A], 0, float('inf'), None)
    return best_cost, best_path


def approximated_path_visiting_every_vertex(graph, A):
    dists, parents = shortest_path_bfs(graph, A)
    # Armamos la lista de remaining con los mas cercanos primero
    remaining = sorted([x for x in graph.vertices() if x != A], key=lambda x: dists[x])
    path = [A]
    total_cost = 0

    while remaining:
        last = remaining.pop()
        costs, parents = dijkstra(graph, path[-1])
        shortest = _reconstruct_path(parents, last)

        for vertex in shortest:
            if vertex in remaining:
                remaining.remove(vertex)

        path += shortest[1:]
        total_cost += costs[last]

    return total_cost, path


def find_n_cycle(graph, n, A):
    # La idea es generar todos lo caminos de n vertices que sale de A y ver cuales forman un ciclo

    def _find_n_cycle(path, visited):

        if len(path) == (n+1):
            if path[-1] == A:
                return path
            return None

        for w in graph.adjacent(path[-1]):
            if w in visited:
                continue
            visited.add(w)
            result = _find_n_cycle(path + [w], visited)
            visited.remove(w)
            if result:
                return result

        return None

    for v in graph.adjacent(A):
        starting_path = [v]
        cycle = _find_n_cycle(starting_path, set())
        if cycle:
            return [A] + cycle
    return None


def topological_sort(graph):
    degrees = collections.defaultdict(int)

    # O(V + E)
    for v in graph.vertices():
        for w in graph.adjacent(v):
            degrees[w] += 1

    # O(V + E)
    partial_order = []
    queue = collections.deque([v for v in graph.vertices() if degrees[v] == 0])
    while queue:
        v = queue.pop()
        for w in graph.adjacent(v):
            degrees[w] -= 1
            if degrees[w] == 0:
                queue.appendleft(w)
        partial_order.append(v)
    return partial_order


def export_kml(path, position_map):
    final_text = []

    def write(final_text, text, indent_level=0):
        final_text.append(' ' * 4 * indent_level + text + '\n')

    def write_placemark(final_text, placemark_type, coordinates_text, indent_level=0, name=None):
        write(final_text, '<Placemark>', indent_level=0 + indent_level)
        if name:
            write(final_text, f'<name>{vertex}</name>', indent_level=1 + indent_level)
        write(final_text, f'<{placemark_type}>', indent_level=1 + indent_level)
        write(final_text, f'<coordinates>{coordinates_text}</coordinates>', indent_level=2 + indent_level)
        write(final_text, f'</{placemark_type}>', indent_level=1 + indent_level)
        write(final_text, '</Placemark>', indent_level=0 + indent_level)
        write(final_text, '', indent_level=1 + indent_level)

    write(final_text, '<?xml version="1.0" encoding="UTF-8"?>')
    write(final_text, '<kml xmlns="http://www.opengis.net/kml/2.2">')
    write(final_text, '<Document>', indent_level=1)
    write(final_text, '<name>KML TP3</name>', indent_level=2)
    for vertex in path:
        lat, long = position_map[vertex]
        write_placemark(final_text, 'Point', f'{lat}, {long}', indent_level=2, name=vertex)

    for i in range(len(path)-1):
        v = path[i]
        w = path[i+1]
        latv, longv = position_map[v]
        latw, longw = position_map[w]
        write_placemark(final_text, 'LineString', f'{latv}, {longv} {latw}, {longw}', indent_level=2)

    write(final_text, '</Document>', indent_level=1)
    write(final_text, '</kml>', indent_level=0)

    return ''.join(final_text)
