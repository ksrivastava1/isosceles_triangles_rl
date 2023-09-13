from numba import cuda, float32, int32
import numpy as np

Lambda = 0.5
board_construction = 'spiral'
board_type = 'torus'

@cuda.jit
def create_coordinates(n, coordinates):
    i, j = cuda.grid(2)
    if i < n and j < n:
        coordinates[i * n + j, 0] = i
        coordinates[i * n + j, 1] = j

@cuda.jit
def generate_distance_matrix(n, board_type, dist_matrix):
    i, j = cuda.grid(2)
    if i < n * n and j < n * n:
        if board_type == 0:  # euclidean
            dx = coordinates[i, 0] - coordinates[j, 0]
            dy = coordinates[i, 1] - coordinates[j, 1]
            dist_matrix[i, j] = cuda.sqrt(dx * dx + dy * dy)
            dist_matrix[j, i] = dist_matrix[i, j]
        elif board_type == 1:  # torus
            row1, col1 = divmod(i, n)
            row2, col2 = divmod(j, n)
            dx = cuda.abs(col2 - col1)
            dx = cuda.min(dx, n - dx)
            dy = cuda.abs(row2 - row1)
            dy = cuda.min(dy, n - dy)
            dist_matrix[i, j] = cuda.sqrt(dx * dx + dy * dy)
            dist_matrix[j, i] = dist_matrix[i, j]

@cuda.jit(device=True)
def torus_distance(i1, j1, i2, j2, rows, cols):
    d1 = cuda.abs(j2 - j1)
    d1 = cuda.min(d1, cols - d1)
    d2 = cuda.abs(i2 - i1)
    d2 = cuda.min(d2, rows - d2)
    return cuda.sqrt(d1 ** 2 + d2 ** 2)

@cuda.jit(device=True)
def count_elements(row, elements):
    i = cuda.grid(1)
    if i < row.size:
        x = row[i]
        cuda.atomic.add(elements, x, 1)

@cuda.jit
def count_isosceles_triangles(board, distance_matrix, p, count):
    i = cuda.grid(1)
    if i < p:
        row = distance_matrix[i]
        hash_table = cuda.shared.array(128, int32)
        for j in range(row.size):
            cuda.atomic.add(hash_table, int(row[j]), 1)
        for j in range(hash_table.size):
            count[0] += hash_table[j] - 1

def main():
    n = 10
    len_word = n * n * 2
    p = 10  # number of ones

    # Create coordinates
    coordinates = cuda.device_array((n * n, 2), dtype=float32)
    threadsperblock = (16, 16)
    blockspergrid_x = (n * n + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n * n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    create_coordinates[blockspergrid, threadsperblock](n, coordinates)

    # Generate distance matrix
    dist_matrix = cuda.device_array((n * n, n * n), dtype=float32)
    blockspergrid_x = (n * n + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n * n + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    generate_distance_matrix[blockspergrid, threadsperblock](n, board_type, dist_matrix)

    # Count isosceles triangles
    board = cuda.to_device(np.random.randint(0, 2, size=(n, n)).astype(np.int32))
    count = cuda.device_array(1, dtype=int32)
    count[0] = 0
    count_isosceles_triangles[1, p](board, dist_matrix, p, count)

    # Print the count
    cuda.synchronize()
    print("Count of isosceles triangles:", count[0])

if __name__ == '__main__':
    main()

