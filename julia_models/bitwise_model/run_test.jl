include("bitwise_model.jl")

if Threads.nthreads() > 1 
    BLAS.set_num_threads(1)   # this seems to help a bit
end

function main()
    n_low = 5  # board size lower bound (pick a board size above 3)
    n_up = 10   # board size upper bound (pick a board size above 3)
    write_all = false
    write_best = true
    slow = true # This is for the manual counting method which is more space efficient but slower

    for i in n_low:n_up
        filename = "$(i)x$(i)"
        train(i, write_all, write_best, filename, slow)
    end
end

main()
