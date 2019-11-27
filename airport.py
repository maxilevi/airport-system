import sys
from graph import Graph
import graph_lib
import collections

# Operaciones


def list_operations(airports, flights, args):
    for x in build_command_map().keys():
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
    graph = _build_graph(airports, flights, lambda x: x['flight_count'])
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


def pagerank(airports, flights, args):
    return


def new_airline(airports, flights, args):
    out_file = args[0]
    graph = _build_graph(airports, flights, weight_func=lambda f: f['price'], is_undirected=True)
    mst = graph_lib.build_MST(graph)

    flight_map = {}
    for flight in flights:
        flight_map[(flight['i'], flight['j'])] = flight

    def write_if_exists(edge):
        if edge in flight_map:
            flight = flight_map[edge]
            f.write(f"{flight['i']},{flight['j']},{flight['avg_time']},{flight['price']},{flight['flight_count']}\n")

    with open(out_file, 'w') as f:
        for v, w, _ in mst.edges():
            write_if_exists((v, w))
    print('OK')


def world_tour(airports, flights, args):
    return


def approximated_world_tour(airports, flights, args):
    return


def vacations(airports, flights, args):
    return


def schedule(airports, flights, args):
    return


def export_kml(airports, flights, args):
    out_path = args[0]
    flight_path = args[1]
    airport_map = _build_airport_position_map(airports)

    with open(out_path, 'w') as f:
        f.write(graph_lib.export_kml(flight_path, airport_map))

# Auxiliaries


def _build_graph(airports, flights, weight_func=lambda x: 1, is_undirected=False):
    graph = Graph(is_undirected=is_undirected)
    for airport in airports:
        graph.add_vertex(airport['code'])
    for flight in flights:
        graph.add_edge((flight['i'], flight['j']), weight=weight_func(flight))
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
        'pagerank': pagerank,
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
    name = args[0]
    arguments = [x.strip() for x in ' '.join(args[1:]).split(',')]
    # Esto es como un hack porque si lo paso a todos los comando no puedo hacer unpacking e.g. x, y = args
    if 'exportar_kml' == name:
        arguments += [last_path]
    return build_command_map()[name](airports, flights, arguments)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        airports_path = sys.argv[1]
        flights_path = sys.argv[2]

        with open('./comandos.txt', 'r') as f:
            last_path = None
            for line in f.readlines():
                path = execute_command(line, load_airports(airports_path), load_flights(flights_path), last_path)
                if path is not None: last_path = path
    else:
        print(f"Uso correcto: ./{sys.argv[0]} <archivo_aeropuertos> <archivo_vuelos>")