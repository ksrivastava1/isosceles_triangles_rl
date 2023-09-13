#########################
# UNCOMMENT THESE IF RUNNING THIS BY ITSELF
# OTHERWISE, THE MAIN "bitwise_model.jl" WILL ALREADY HAVE 
# THESE INCLUDED

# using Random
# using Base.Threads
# using Base.Iterators: product

# Lambda = 0.5 # Weight for regularizing the reward function to generate more ones (too high a labda will result in higher odds of generating isosceles triangles)
# board_type = "euclidean" # "euclidean" or "torus" depending on the embedding
# board_construction = "rowwise" # 'rowwise' or 'spiral'
# n = 3

#########################

n = 0


function update_n(board_size)
    global n
    n = board_size
    global POINT_SET
    POINT_SET = all_points()
end

# Helper function for computing torus distance for later

function torus_distance(a,b) # a and b are tuples of coordinates
    return min(abs(a[1]-b[1]), n-abs(a[1]-b[1]))^2 + min(abs(a[2]-b[2]), n-abs(a[2]-b[2]))^2
end

function dist(a,b)
    return (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])
end

function create_coordinates(n)
    coordinates = zeros(Float64, n^2, 2)
    @threads for i in 1:n
        for j in 1:n
            coordinates[(i - 1) * n + j, 1] = i - 1
            coordinates[(i - 1) * n + j, 2] = j - 1
        end
    end
    return coordinates
end

function all_points()
    points = Vector{Tuple{Int64, Int64}}(undef, 0)
    for i in 1:n
      for j in 1:n
        if i <= n  || j <= n
          append!(points,[(i,j)])
        end
     end
    end
    return points
end

function isosceles(a::Tuple{Int64, Int64},b::Tuple{Int64, Int64},c::Tuple{Int64, Int64})
    if dist(a,b) == dist(a,c)
      return true
    end
    if dist(a,b) == dist(b,c)
      return true
    end
    if dist(a,c) == dist(b,c)
      return true
    end
    return false
end

function generate_distance_matrix(n)
    if board_type == "euclidean"
        # Create a meshgrid of the points
        coordinates = zeros(Float32, n * n, 2)
        for i in 1:n
            for j in 1:n
                coordinates[(i - 1) * n + j, 1] = i - 1
                coordinates[(i - 1) * n + j, 2] = j - 1
            end
        end

        # Compute pairwise Euclidean distances between points
        dist_matrix = zeros(Float32, n * n, n * n)
        for i in 1:(n * n)
            for j in i:(n * n)
                dx = coordinates[i, 1] - coordinates[j, 1]
                dy = coordinates[i, 2] - coordinates[j, 2]
                dist_matrix[i, j] = sqrt(dx * dx + dy * dy)
                dist_matrix[j, i] = dist_matrix[i, j]  # Add to the transpose
            end
        end
    elseif board_type == "torus"
        # Compute pairwise distances between points using torus distance
        dist_matrix = zeros(Float32, n * n, n * n)
        for i in 1:(n * n)
            for j in i:(n * n)
                row1, col1 = divrem(i - 1, n)  # Convert to 0-based indexing
                row2, col2 = divrem(j - 1, n)  # Convert to 0-based indexing
                dx = abs(col2 - col1)
                dx = min(dx, n - dx)
                dy = abs(row2 - row1)
                dy = min(dy, n - dy)
                dist_matrix[i, j] = sqrt(dx * dx + dy * dy)
                dist_matrix[j, i] = dist_matrix[i, j]  # Add to the transpose
            end
        end
    else
        throw(ArgumentError("Unsupported board type: $board_type"))
    end

    return dist_matrix
end

function state_to_word(state, len_word)
    word = zeros(Float32, len_word)
    for i in 1:len_word
        word[i] = state[i]
    end
    for i in len_word+1:2*len_word
        if state[i] == 1
            word[i - len_word] = 1
        end
    end
    return word
end

function convert_to_board(word, n)
    # word = convert(Vector{Float32}, word)  # Cast input array to Float32
    board = zeros(Float32, n, n)
    
    if board_construction == "rowwise"
        for i in 1:length(word)
            board[(i - 1) ÷ n + 1, (i - 1) % n + 1] = word[i]
        end
    elseif board_construction == "spiral"
        i, j = 1, 1  # Starting coordinates
        direction = "right"
        visited = zeros(Float32, n, n)
        
        for k in 1:length(word)
            board[i, j] = word[k]
            visited[i, j] = 1
            
            if direction == "right"
                if j + 1 ≤ n && visited[i, j + 1] == 0
                    j += 1
                else
                    direction = "down"
                    i += 1
                end
            elseif direction == "down"
                if i + 1 ≤ n && visited[i + 1, j] == 0
                    i += 1
                else
                    direction = "left"
                    j -= 1
                end
            elseif direction == "left"
                if j - 1 ≥ 1 && visited[i, j - 1] == 0
                    j -= 1
                else
                    direction = "up"
                    i -= 1
                end
            elseif direction == "up"
                if i - 1 ≥ 1 && visited[i - 1, j] == 0
                    i -= 1
                else
                    direction = "right"
                    j += 1
                end
            end
        end
    end
    
    return board
end

function count_elements(row)
    # Initialize an empty dictionary (Julia equivalent: Dict)
    d = Dict{Any, Int}()

    # Loop over each element in the row
    for x in row
        # If the element is already in the dictionary, increment its count
        if haskey(d, x)
            d[x] += 1
        # Otherwise, add it to the dictionary with a count of 1
        else
            d[x] = 1
        end
    end

    # Return the resulting dictionary
    return d
end

function count_isosceles_triangles(board, distance_matrix, p)
    # Find indices of all ones on the board
    one_indices = findall(board .== 1)
    
    # Compute pairwise distances between all ones and store in a matrix
    # So row i is the distances from the ith one to all other ones
    ones_distance = zeros(Float32, p, p)

    for i in 1:p
        for j in i+1:p
            ones_distance[i, j] = distance_matrix[one_indices[i], one_indices[j]]
            ones_distance[j, i] = ones_distance[i, j]
        end
    end

    # Initialize count of isosceles triangles to zero
    count = 0

    # Check how many repeated distances there are in each row
    # Each repeated distance is an isosceles triangle
    for i in 1:p
        row = ones_distance[i, :]
        hash_table = count_elements(row)
        for key in keys(hash_table)
            temp = hash_table[key] - 1
            if temp == 0 || temp == 1
                count += temp
            else
                temp = Int((temp+1) * (temp + 2) / 2)
                count += temp
            end
        end
    end

    return count
end

function get_score(word, dist_matrix, len_word, n, slow)

    # Cast input array to float32
    word = Float32.(word)

    # If the size of the word is 2*len_word, then convert it to a word first
    # We need this because in this case, the second half of 'word' contains the information of the one we
    # are considering placing on the board. This places it into the word and we then convert it to a board.
    if length(word) == 2 * len_word
        word = state_to_word(word, len_word)
    end

    board = convert_to_board(word, n)

    # Get the number of 1s on the board
    p = sum(board)

    if p >= 3 * n
        return -10000
    end

    # Calculate the number of isosceles triangles on the board
    if slow
        score = reward_calc(word)
    else
        isosceles = count_isosceles_triangles(board, dist_matrix, Int(p))
    end

    return score 
end

function reward_calc(obj)::Float32
    penalty::Int64 = 0
    points = Vector{Tuple{Int64, Int64}}(undef, 0)
    counter::Int64 = 1
    for (i,j) in POINT_SET::Vector{Tuple{Int64, Int64}}
        if obj[counter] == 1
            push!(points,(i,j))
        end
        counter += 1
    end

    for i in 1:length(points)
        for j in i+1:length(points)
            for k in j+1:length(points)
                if isosceles(points[i],points[j],points[k])
                    penalty += 1
                end
            end
        end
    end

    return (-1* penalty) + (Lambda * sum(obj))
end
