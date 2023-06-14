# CODE HERE PROVIDED BY ADAM Z. WAGNER
# You can find more information by their work in the following link:
# https://users.wpi.edu/~zadam/index.html, https://github.com/zawagner22?tab=repositories
# Read about his work in applying Reinforcement learning to Mathematics for more details and cool information!

using Base.Threads
using Random
using LinearAlgebra


const N::Int64 = 64

const ENTRY_TYPE = UInt8   # UInt8 should be more memory efficient in theory
const OBJ_TYPE = Vector{ENTRY_TYPE}  # type for vectors encoding constructions
const REWARD_TYPE = Int64  # type for rewards 


if Threads.nthreads() > 1 
    BLAS.set_num_threads(1)   # this seems to help a bit
end

println("Using ", nthreads(), " thread(s)")

function all_points()
  points = Vector{Tuple{Int64, Int64}}(undef, 0)
  for i in 1:N
    for j in 1:N
      if i <= N  || j <= N
        append!(points,[(i,j)])
      end
   end
  end
  return points
end

const POINT_SET::Vector{Tuple{Int64, Int64}} = all_points()

function dist(a,b)
    return (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2])
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

function reward_calc(obj::OBJ_TYPE)::Int64
    penalty::Int64 = 0
    points = Vector{Tuple{Int64, Int64}}(undef, 0)
    counter::Int64 = 1
    for (i,j) in POINT_SET::Vector{Tuple{Int64, Int64}}
        if obj[counter] == 2
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

    return length(points) - penalty
end


function read_grid()
    res = Vector{ENTRY_TYPE}[]
    open("input_grid.txt") do f       
        count = 1  
        x = zeros(ENTRY_TYPE, N*N)
        for a in 1:N
            s = readline(f) 
            for i in 1:N
                if s[i] == '.'
                    x[count] = 1
                elseif s[i] == 'o'
                    x[count] = 2
                else
                    println("BAD")
                    println(s[i])
                    exit()
                end
                count += 1
            end
        end  
        push!(res, copy(x))
    end
    return res
end


function rew_calc_many_times(grid, repeat_count; multithreaded=false)
    rew::REWARD_TYPE = 0
    if !multithreaded
        for _ in 1:repeat_count
            rew = reward_calc(grid)
        end
        return rew
    else
        @threads for _ in 1:repeat_count
            rew = reward_calc(grid)
        end
        return rew
    end
end

function main()
    grid = read_grid()[1]
    rew = rew_calc_many_times(grid, 1)                     #Have to run it once to compile the code 
    rew = rew_calc_many_times(grid, 1, multithreaded=true) #Have to run it once to compile the code 
    println("Reward is: " * string(rew))

    print("Single threaded: ")
    @time rew_calc_many_times(grid, 1000)
    print("Multithreaded: ")
    @time rew_calc_many_times(grid, 1000, multithreaded=true)
end

main()