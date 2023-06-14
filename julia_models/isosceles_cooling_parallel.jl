# CODE HERE PROVIDED BY ADAM Z. WAGNER
# You can find more information by their work in the following link:
# https://users.wpi.edu/~zadam/index.html, https://github.com/zawagner22?tab=repositories
# Read about his work in applying Reinforcement learning to Mathematics for more details and cool information!


using Random
using LinearAlgebra
using Statistics
using StatsBase
using Plots
using Base.Threads

type = "euclidean" # "euclidean" or "torus" depending on the embedding

for N in 32:32

    STEP_SIZE::Float64 = 0.11


    function torus_dist(a, b)
        d1 = min(abs(b[2] - a[2]), N - abs(b[2] - a[2]))
        d2 = min(abs(b[1] - a[1]), N - abs(b[1] - a[1]))
        return (d1^2 + d2^2)
    end

    function dist(a,b)::Int64
        if type == "torus"
            return torus_dist(a,b)
        end
        if type == "euclidean"
            return (a[1]-b[1])^2 + (a[2]-b[2])^2
        end
    end


    function get_all_isosceles_triangles_in_the_grid()
        same_dist_points = Dict{Int64,Vector{Tuple{Int64,Int64}}}()
        distances = Dict{Int64,Vector{Tuple{Int64,Int64}}}()
        for j1 in -N:N
            for j2 in -N:N
                d = dist((0,0),(j1,j2))
                if haskey(distances,d)
                    push!(distances[d],(j1,j2))
                else
                    distances[d] = [(j1,j2)]
                end
            end
        end
        for (key, value) in distances
            same_dist_points[key] = value
        end

        return same_dist_points
    end

    same_dist_points::Dict{Int64,Vector{Tuple{Int64,Int64}}} = get_all_isosceles_triangles_in_the_grid()


    function valid(a::Tuple{Int64,Int64},b::Tuple{Int64,Int64})::Bool
        if a[1]+b[1] < 1 || a[1]+b[1] > N || a[2]+b[2]<1 || a[2]+b[2]>N
            return false
        end
        return true
    end


    function evaluate_grid_at_point(grid::Matrix{Float64}, point::Tuple{Int64,Int64})::Float64
        answer::Float64 = 0
        for item in same_dist_points
            a::Float64 = 0
            for relpt in item[2]
                if valid(relpt,point)
                    a+=grid[relpt[1]+point[1], relpt[2]+point[2]]
                end
            end
            answer += a^2

            b::Float64 = 0
            for relpt in item[2]
                if valid(relpt,point)
                    b+=grid[relpt[1]+point[1], relpt[2]+point[2]]^2
                end
            end
            answer -= b 
            if answer > 2.0
                return 10.0
            end
        end
        answer /= 2.0

        for i1::Int64 in 1:N
            for i2::Int64 in 1:N
                item = same_dist_points[dist(point, (i1,i2))]
                for relpt in item
                    if valid(relpt,(i1,i2)) && (relpt[1]+i1,relpt[2]+i2) != point
                        answer += grid[i1,i2] * grid[relpt[1]+i1,relpt[2]+i2]
                        if answer > 1.0
                            return 10.0
                        end
                    end
                end
            end
        end

        return answer
    end


    function initialize_grid()::Matrix{Float64}
        return rand(Float64,N,N)
        grid = Matrix{Float64}(undef,N,N)
        for i in 1:N
            for j in 1:N
                if rand()<0.98
                    grid[i,j] = 0
                else
                    grid[i,j] = rand()
                end
                grid[i,j] = rand()/4.0
            end
        end
        return grid
    end


    function print_nicely(grid)
        adjmat = zeros(Int8, N,N)
        for i in 1:N
            for j in 1:N
                if grid[i,j] > 0.8
                    adjmat[i,j] = 1
                end
            end
        end
        f = open("ellenberg_geordie_method_"*string(N)*"_"*string(sum(adjmat))*".txt", "w")
        for i in 1:N
            for j in 1:N
            if adjmat[i,j] == 1
                write(f,"0")
            else
                write(f,"·")
            end
            end
            write(f,"\n")
        end
        write(f,"\n\n\n\n")
        close(f)
    end


    function check_if_cooled(grid)::Bool
        for i in 1:N
            for j in 1:N
                if grid[i,j] > 0.01 && grid[i,j] < 0.99
                    return false
                end
            end
        end
        return true
    end


    function main_greedy_loop(grid)
        for step in 1:100_000
            @threads for pointx::Int64 in 1:N
                for pointy::Int64 in 1:N
                    direction::Float64 = evaluate_grid_at_point(grid,(pointx,pointy))
                    if direction >= 1 && grid[pointx,pointy] > 0
                        grid[pointx,pointy] = max(0,grid[pointx,pointy] - STEP_SIZE)
                    elseif direction < 1 && grid[pointx,pointy] < 1
                        grid[pointx,pointy] = min(1,grid[pointx,pointy] + STEP_SIZE)
                    end
                end
            end
            print(".")
            gr()
            if !isdir("heatmaps_"*string(N))
                mkdir("heatmaps_"*string(N))
            end
            savefig(heatmap(1:size(grid,1),
                1:size(grid,2), grid,
                c=cgrad([:white, :black]), size=(850,750),
                clim=(0,1.0)), "heatmaps_"*string(N)*"/"*string(step)*".png")

            if check_if_cooled(grid)
                break
            end
        end
        print_nicely(grid)
    end


    function main()
        println("All triangles have been calculated.")
        grid = initialize_grid()
        time_main = @elapsed main_greedy_loop(grid)
        println("Time elapsed: " * string(time_main))
    end

    main()
end