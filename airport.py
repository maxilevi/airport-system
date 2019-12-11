#!/usr/bin/env python3

import sys
from graph import Graph
import graph_lib
import collections

# Operaciones


def list_operations(airports, flights, args):
    for x in build_command_map().keys():
        # Medio un hack pero bueno
        if not x == 'listar_operaciones':
            print(x)


def _base_best_path(src, dst, airports, flights, calc_weight_func):

    # En el caso que el grafo sea 'unweighted' esto va a sumar 1 por cada edge, dando siempre el camino mas corto.
    def calc_heuristic(curr_path):
        val = 0
        for i in range(len(curr_path)-1):
            val += graph.weight((curr_path[i], curr_path[i+1]))
        return val

    graph = _build_graph(airports, flights, calc_weight_func)
    city_airport_map = _build_city_airport_map(airports)
    best_val = float('inf')
    best = None

    # Hacemos todas las combinaciones de los aeropuertos de cada ciudad y nos guardamos el mejor
    # Como en la consigna dice que la cantidad es depreciable, el orden deberia ser igual.

    for src_airport in city_airport_map[src]:
        for dst_airport in city_airport_map[dst]:
            path = graph_lib.shortest_path(graph, src_airport, dst_airport)
            if not path: continue
            heuristic = calc_heuristic(path)
            if heuristic < best_val:
                best = path
                best_val = heuristic

    display_path(best)
    return best


def best_path(airports, flights, args):
    option, src, dst = args

    def calc_weight(flight):
        return flight['price'] if option == 'barato' else flight['avg_time']

    return _base_best_path(src, dst, airports, flights, calc_weight)


def less_stops(airports, flights, args):
    src, dst = args
    # Pasamos una lambda que le ponga 1 de weight a todos los vertices, equivalente a q sea 'unweighted'
    return _base_best_path(src, dst, airports, flights, lambda x: 1)


def centrality(airports, flights, args):
    graph = _build_graph(airports, flights, lambda x: 1.0 / x['flight_count'])
    n = int(args[0])
    display_centrality(
        graph_lib.betweeness_centrality(graph),
        n
    )


def approximated_centrality(airports, flights, args):
    graph = _build_graph(airports, flights, lambda x: x['flight_count'])
    n = int(args[0])
    display_centrality(
        graph_lib.approximate_centrality(graph),
        n
    )


def new_airline(airports, flights, args):
    out_file = args[0]
    graph = _build_graph(airports, flights, weight_func=lambda f: f['price'])
    mst = graph_lib.build_MST(graph)

    flight_map = {}
    for flight in flights:
        flight_map[(flight['i'], flight['j'])] = flight
        flight_map[(flight['j'], flight['i'])] = flight

    with open(out_file, 'w') as f:
        for v, w, weight in mst.edges():
            flight = flight_map[(v, w)]
            f.write(f"{flight['i']},{flight['j']},{flight['avg_time']},{flight['price']},{flight['flight_count']}\n")
    print('OK')


def _base_world_tour(airports, flights, args, path_builder):
    src = args[0]
    city_airport_map = _build_city_airport_map(airports)
    graph = _build_graph(airports, flights, weight_func=lambda x: x['avg_time'])
    min_cost = float('inf')
    min_path = None
    for airport in city_airport_map[src]:
        curr_min_cost, curr_min_path = path_builder(graph, airport)
        if curr_min_path and curr_min_cost < min_cost:
            min_cost = curr_min_cost
            min_path = curr_min_path
    display_path(min_path)
    print(f'Costo: {min_cost}')
    return min_path


def world_tour(airports, flights, args):
    return _base_world_tour(airports, flights, args, graph_lib.path_visiting_every_vertex)


def approximated_world_tour(airports, flights, args):
    return _base_world_tour(airports, flights, args, graph_lib.approximated_path_visiting_every_vertex)


def vacations(airports, flights, args):
    src, n = args
    city_airport_map = _build_city_airport_map(airports)
    graph = _build_graph(airports, flights)

    path = None
    for airport in city_airport_map[src]:
        path = graph_lib.find_n_cycle(graph, int(n), airport)
        if path:
            break

    if not path:
        print('No se encontro recorrido')
    else:
        display_path(path)
    return path


def schedule(airports, flights, args):
    file_path = args[0]
    restrictions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        cities = [x.strip() for x in lines[0].split(',')]
        for i in range(1, len(lines)):
            i, j = [x.strip() for x in lines[i].split(',')]
            restrictions.append((i, j))

    graph = Graph()
    for city in cities:
        graph.add_vertex(city)
    for edge in restrictions:
        graph.add_edge(edge)

    sorted_vertices = graph_lib.topological_sort(graph)
    print(', '.join(sorted_vertices))
    for i in range(len(sorted_vertices)-1):
        less_stops(airports, flights, [sorted_vertices[i], sorted_vertices[i+1]])


def export_kml(airports, flights, args):
    out_path = args[0]
    flight_path = args[1]
    airport_map = _build_airport_position_map(airports)

    with open(out_path, 'w') as f:
        f.write(graph_lib.export_kml(flight_path, airport_map))

# Auxiliaries


def _build_graph(airports, flights, weight_func=lambda x: 1, is_undirected=True):
    graph = Graph(is_undirected=is_undirected)
    for airport in airports:
        graph.add_vertex(airport['code'])
    for flight in flights:
        edge = (flight['i'], flight['j'])
        graph.add_edge(edge, weight=weight_func(flight))
    return graph


def _build_airport_position_map(airports):
    airport_map = {}
    for entry in airports:
        airport_map[entry['code']] = (entry['latitude'], entry['longitude'])
    return airport_map


def _build_city_airport_map(airports):
    city_map = collections.defaultdict(list)
    for entry in airports:
        city_map[entry['city']].extend([entry['code']])
    return city_map


def display_centrality(centrality, n):
    sorted_codes = sorted(centrality.keys(), key=lambda x: -centrality[x])
    print(', '.join(sorted_codes[:n]))


def display_path(path):
    print(' -> '.join(path))

# Boilerplate


def build_command_map():
    return {
        'listar_operaciones': list_operations,
        'camino_mas': best_path,
        'camino_escalas': less_stops,
        'centralidad': centrality,
        'centralidad_aprox': approximated_centrality,
        'nueva_aerolinea': new_airline,
        'recorrer_mundo': world_tour,
        'recorrer_mundo_aprox': approximated_world_tour,
        'vacaciones': vacations,
        'itinerario': schedule,
        'exportar_kml': export_kml
    }


def load_file(path, parser):
    contents = []
    with open(path, 'r') as f:
        split_lines = [x.split(',') for x in f.readlines()]
        contents.extend(
            parser(split_lines)
        )
    return contents


def load_airports(path):
    def parse(split_lines):
        return [{'city': city, 'code': code, 'latitude': float(lat), 'longitude': float(long)} for city, code, lat, long in split_lines]

    return load_file(path, parse)


def load_flights(path):
    def parse(split_lines):
        return [{'i': i, 'j': j, 'avg_time': int(k), 'price': int(l), 'flight_count': int(m)} for i, j, k, l, m in split_lines]
    return load_file(path, parse)


def execute_command(line, airports, flights, last_path):
    args = line.split(' ')
    name = args[0].strip()
    arguments = [x.strip() for x in ' '.join(args[1:]).split(',')]
    # Esto es algo medio feo pero si lo paso a todos los comandos no puedo hacer unpacking e.g. x, y = args
    if 'exportar_kml' == name:
        arguments += [last_path]
    return build_command_map()[name](airports, flights, arguments)


def execute():
    if len(sys.argv) == 3:
        airports_path = sys.argv[1]
        flights_path = sys.argv[2]

        last_path = None
        for line in sys.stdin:
            path = execute_command(line, load_airports(airports_path), load_flights(flights_path), last_path)
            if path is not None: last_path = path
    else:
        print(f"Uso correcto: ./{sys.argv[0]} <archivo_aeropuertos> <archivo_vuelos>")


if __name__ == "__main__":
    execute()
