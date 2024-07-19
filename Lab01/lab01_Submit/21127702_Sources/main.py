# Fullname:     Bui Nguyen Tin
# ID Student:   21127702

from collections import deque
import time
import tracemalloc
import numpy as np
import heapq


# 1. Search Strategies Implementation
# 1.1. Breadth-first search (BFS)
def bfs(arr, source, destination):
    # TODO
    path = []
    visited = {}
    queue = deque([(source, [source])])

    while queue:
        current, path = queue.popleft()

        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        for neighbor, connected in enumerate(arr[current]):
            if connected and neighbor not in visited:
                visited[neighbor] = current
                queue.append((neighbor, path + [neighbor]))
    
    return {}, []


# 1.2. Depth-first search (DFS)
def dfs(arr, source, destination):
    # TODO
    path = []
    visited = {}
    stack = [(source, [source])]

    while stack:
        current, path = stack.pop()

        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        for neighbor, connected in enumerate(arr[current]):
            if connected and neighbor not in visited:
                visited[neighbor] = current
                stack.append((neighbor, path + [neighbor]))

    return {}, []


# 1.3. Uniform-cost search (UCS)
def ucs(arr, source, destination):
    # TODO
    path = []
    visited = {}
    queue = [(0, source, [source])]
    heapq.heapify(queue)

    while queue:
        cost, current, path = heapq.heappop(queue)

        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        if current not in visited:
            visited[current] = path[-2] if len(path) > 1 else None
            for neighbor, weight in enumerate(arr[current]):
                if weight > 0 and neighbor not in visited:
                    heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))

    return {}, []


# 1.4. Iterative deepening search (IDS)
# 1.4.a. Depth-limited search
def dls(arr, source, destination, depth_limit):
    # TODO
    def recursive_dls(current, destination, depth, path, visited):
        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        if depth == 0:
            return {}, []

        for neighbor, connected in enumerate(arr[current]):
            if connected and neighbor not in visited:
                visited[neighbor] = current
                result_visited, result_path = recursive_dls(neighbor, destination, depth - 1, path + [neighbor], visited)
                if result_path:
                    return result_visited, result_path

        return {}, []

    return recursive_dls(source, destination, depth_limit, [source], {source: None})


# 1.4.b. IDS
def ids(arr, source, destination):
    # TODO
    depth = 0
    while True:
        visited, path = dls(arr, source, destination, depth)
        if path:
            return visited, path
        depth += 1


# 1.5. Greedy best first search (GBFS)
def gbfs(arr, source, destination, heuristic):
    # TODO
    path = []
    visited = {}
    queue = [(heuristic[source], source, [source])]
    heapq.heapify(queue)

    while queue:
        _, current, path = heapq.heappop(queue)

        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        if current not in visited:
            visited[current] = path[-2] if len(path) > 1 else None
            for neighbor, connected in enumerate(arr[current]):
                if connected and neighbor not in visited:
                    heapq.heappush(queue, (heuristic[neighbor], neighbor, path + [neighbor]))

    return {}, []


# 1.6. Graph-search A* (AStar)
def astar(arr, source, destination, heuristic):
    # TODO
    path = []
    visited = {}
    queue = [(heuristic[source], 0, source, [source])]
    heapq.heapify(queue)

    while queue:
        _, cost, current, path = heapq.heappop(queue)

        if current == destination:
            for i in range(len(path) - 1):
                visited[path[i+1]] = path[i]
            return visited, path

        if current not in visited:
            visited[current] = path[-2] if len(path) > 1 else None
            for neighbor, weight in enumerate(arr[current]):
                if weight > 0 and neighbor not in visited:
                    heapq.heappush(queue, (cost + weight + heuristic[neighbor], cost + weight, neighbor, path + [neighbor]))

    return {}, []


# 1.7. Hill-climbing First-choice (HC)
def hc(arr, source, destination, heuristic):
    # TODO
    
    path = [source]
    visited = {source: None}
    current = source

    while current != destination:
        neighbors = [(heuristic[neighbor], neighbor) for neighbor, connected in enumerate(arr[current]) if connected and neighbor not in visited]
        if not neighbors:
            return {}, []
        _, next_node = min(neighbors)
        if heuristic[next_node] >= heuristic[current]:
            return {}, []
        visited[next_node] = current
        path.append(next_node)
        current = next_node

    return visited, path


# Functions for main
def Measure(algorithm, arr, source, destination, heuristic=None, depth_limit=None):
    start_time = time.perf_counter()
    tracemalloc.start()
    if algorithm == "bfs":
        visited, path = bfs(arr, source, destination)
    elif algorithm == "dfs":
        visited, path = dfs(arr, source, destination)
    elif algorithm == "ucs":
        visited, path = ucs(arr, source, destination)
    elif algorithm == "dls":
        visited, path = dls(arr, source, destination, depth_limit)
    elif algorithm == "ids":
        visited, path = ids(arr, source, destination)
    elif algorithm == "gbfs":
        visited, path = gbfs(arr, source, destination, heuristic)
    elif algorithm == "astar":
        visited, path = astar(arr, source, destination, heuristic)
    elif algorithm == "hc":
        visited, path = hc(arr, source, destination, heuristic)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()
    time_taken = format(end_time - start_time, '.10f')
    return {
        'path': path,
        'time': time_taken,
        'memory': peak / 1024
    }


def readInputFile(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    source, destination = map(int, lines[1].strip().split())
    arr = np.array([list(map(int, line.strip().split())) for line in lines[2:n+2]])
    heuristic = list(map(int, lines[n+2].strip().split()))
    return n, source, destination, arr, heuristic


def writeOutputFile(file_path, results):
    with open(file_path, 'w') as f:
        for algorithm, result in results.items():
            f.write(f"{algorithm}:\n")
            if result['path']:
                f.write(f"Path: {' -> '.join(map(str, result['path']))}\n")
            else:
                f.write("Path: -1\n")
            f.write(f"Time: {result['time']} seconds\n")
            f.write(f"Memory: {result['memory']} KB\n\n")


def runSearchAlgorithms(arr, source, destination, depth_limit=None, heuristic=None):
    results = {}
    results['BFS']      = Measure('bfs', arr, source, destination)
    results['DFS']      = Measure('dfs', arr, source, destination)
    results['UCS']      = Measure('ucs', arr, source, destination)
    results['DLS']      = Measure('dls', arr, source, destination, depth_limit=depth_limit)
    results['IDS']      = Measure('ids', arr, source, destination)
    results['GBFS']     = Measure('gbfs', arr, source, destination, heuristic)
    results['AStar']    = Measure('astar', arr, source, destination, heuristic)
    results['HC']       = Measure('hc', arr, source, destination, heuristic)
    
    return results


# 2. Main function
if __name__ == "__main__":
    # TODO: Read the input data
    input_file = input("The input file: ")
    output_file = input("The output file: ")
    
    n, source, destination, arr, heuristic = readInputFile(input_file)
    
        # # Print values from function read_file()
    # print(f"Number of nodes: {n}")
    # print(f"Source node: {source}")
    # print(f"Destination node: {destination}")
    # print("Adjacency matrix:")
    # print(arr)
    # print("Heuristic values:")
    # print(heuristic)

    # TODO: Start measuring
    # TODO: Call a function to execute the path finding process
    # TODO: Stop measuring 
    DepthLimit = 10
    results = runSearchAlgorithms(arr, source, destination, depth_limit = DepthLimit, heuristic = heuristic)

    # TODO: Show the output data
    writeOutputFile(output_file, results)