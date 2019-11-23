import sys
from graph import Graph
import graph_lib

# Operaciones


def list_operations(airports, flights, args):
    for x in build_command_map().keys():
        print(x)


def best_path(airports, flights, args):
    return


def centrality(airports, flights, args):
    graph = _build_centrality_graph(airports, flights)
    n = int(args[0])
    display_centrality(
        graph_lib.betweeness_centrality(graph),
        n
    )


def approximated_centrality(airports, flights, args):
    graph = _build_centrality_graph(airports, flights)
    n = int(args[0])
    display_centrality(
        graph_lib.approximate_centrality(graph),
        n
    )


def pagerank(airports, flights, args):
    return


def new_airline(airports, flights, args):
    return


def world_tour(airports, flights, args):
    return


def approximated_world_tour(airports, flights, args):
    return


def vacations(airports, flights, args):
    return


def schedule(airports, flights, args):
    return


def export_kml(airports, flights, args):
    return

# Auxiliaries


def _build_centrality_graph(airports, flights):
    graph = Graph()
    for airport in airports:
        graph.add_vertex(airport['code'])
    for flight in flights:
        graph.add_edge((flight['i'], flight['j']))
    return graph

def display_centrality(centrality, n):
    return

# Boilerplate


def build_command_map():
    return {
        'listar_operaciones': list_operations,
        'camino_mas': best_path,
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
        return [{'city': city, 'code': code, 'latitude': lat, 'longitude': long} for city, code, lat, long in split_lines]

    return load_file(path, parse)


def load_flights(path):
    def parse(split_lines):
        return [{'i': i, 'j': j, 'avg_time': k, 'price': l, 'flight_count': m} for i, j, k, l, m in split_lines]
    return load_file(path, parse)


def execute_command(line, airports, flights):
    args = line.split(' ')
    name = args[0]
    build_command_map()[name](airports, flights, args[1:])


if __name__ == "__main__":
    if len(sys.argv) == 3:
        airports_path = sys.argv[1]
        flights_path = sys.argv[2]

        with open('./comandos.txt', 'r') as f:
            for line in f.readlines():
                execute_command(line, load_airports(airports_path), load_flights(flights_path))
        #for line in sys.stdin:
        #    execute_command(line, load_airports(airports_path), load_flights(flights_path))
    else:
        print(f"Uso correcto: ./{sys.argv[0]} <archivo_aeropuertos> <archivo_vuelos>")