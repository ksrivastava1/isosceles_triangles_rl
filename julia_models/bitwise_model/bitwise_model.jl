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
using Flux
using Base.Iterators: product
using Flux: Optimise
using Flux: ADAM, params, update!
using Serialization

include("reward_function.jl")

# Variables are for reward_function.jl
Lambda = 0.35 # Weight for regularizing the reward function to generate more ones (too high a labda will result in higher odds of generating isosceles triangles)
board_type = "euclidean" # "euclidean" or "torus" depending on the embedding
board_construction = "rowwise" # 'rowwise' or 'spiral'

# Variables are for bitwise_model.jl
n_actions = 2  # Number of actions that the agent can take. In this case, it is either 0 for excluding a point and 1 for including it
n_sessions = 2000  # Number of new sessions per iteration
learning_rate = 0.001  # Learning rate, increase this to converge faster
percentile = 90  # Top 100-x percentile the agent will learn from
super_percentile = 90  # Top 100-x percentile of that survives to the next generation



########## HELPER FUNCTIONS ##########

# Very Very hacky concatination function
# To fix Julia 3D concatination
# There is probably a nicer way to do this - fix later

function append_2d_to_3d(m3d::Array{Float32, 3}, m2d::Array{Float32, 2})
    m, n, k = size(m3d)
    m_new = m + 1
    result = zeros(Float32, m_new, n, k)
    
    # Copy the original 3D matrix to the new matrix
    result[1:m, :, :] .= m3d
    
    # Append the 2D matrix at the end
    result[m_new, :, :] .= m2d
    
    return result
end

# Another hacky fix to sort a list of a matrix and two vectors 
# based on the last vector

function sort_by_last_vector(m3d::Array{Float32, 3}, v1::Matrix{Int64}, v2::Vector{Float32})
    m, n, k = size(m3d)
    result = zeros(Float32, m, n, k)
    sorted_v1 = zeros(Int64, m, n)
    
    # Sort v2 and find the sorted indices

    sorted_indices = sortperm(v2)
    sorted_v2 = v2[sorted_indices]

    
    for i in 1:m
        sorted_v1[i, :] .= v1[sorted_indices[i], :]
    end
    
    # Copy the original 3D matrix to the new matrix
    for i in 1:m
        result[i, :, :] .= m3d[sorted_indices[i], :, :]
    end
    
    return result, sorted_v1, sorted_v2
end





######### REINFOCEMENT LEARNING FUCNTIONS #########

function generate_session(agent, n_sessions, n, dist_matrix, len_word, len_game, observation_space, verbose=1, slow=true)

    states = zeros(Float32, n_sessions, observation_space, len_game)
    actions = zeros(Int, n_sessions, len_game)
    state_next = zeros(Float32, n_sessions, observation_space)
    prob = zeros(Float32, n_sessions, 1)

    states[:, len_word, 1] .= 1.0

    step = 0
    total_score = zeros(Float32, n_sessions)
    recordsess_time = 0.0
    play_time = 0.0
    scorecalc_time = 0.0
    pred_time = 0.0

    terminal = false

    while !terminal
        step += 1
        tic = time()
        
        for i in 1:n_sessions
            prob[i] = agent(states[i, :, step])[1]
        end

        pred_time += time() - tic

        for i in 1:n_sessions
            if prob[i] > rand()
                action = 1
            else
                action = 0
            end
            actions[i, step] = action

            tic = time()
            state_next[i, :] .= states[i, :, step]
            play_time += time() - tic

            if action > 0
                state_next[i, step] = action
            end

            state_next[i, len_word + step] = 0

            if step < len_word
                state_next[i, len_word + step + 1] = 1
            end

            terminal = step == len_word

            if terminal
                total_score[i] = get_score(state_next[i,:], dist_matrix, len_word, n, slow)
            end

            scorecalc_time += time() - tic
            tic = time()

            if !terminal
                states[i, :, step + 1] .= state_next[i, :]
            end

            recordsess_time += time() - tic
        end

        if terminal
            break
        end
    end

    if verbose == 1
        println("Predict: $pred_time, play: $play_time, scorecalc: $scorecalc_time, recordsess: $recordsess_time")
    end

    return states, actions, total_score
end

function select_elites(states_batch, actions_batch, rewards_batch, percentile)
    counter = n_sessions * (100 - percentile) / 100
    reward_threshold = quantile(rewards_batch, percentile / 100)

    elite_states = Matrix{Float32}(undef, 0, size(states_batch, 3))
    elite_actions = Vector{Int}(undef, 0)

    for i in 1:size(states_batch, 1)
        if rewards_batch[i] >= reward_threshold - 0.000001 && counter > 0
            for item in eachrow(states_batch[i, :, :])
                temp = reshape(item, 1, size(states_batch, 3))
                elite_states = vcat(elite_states, temp)
            end
            for item in actions_batch[i, :]
                push!(elite_actions, item)
            end
        end
        counter -= 1
    end

    return elite_states, elite_actions
end

function select_super_sessions(states_batch, actions_batch, rewards_batch, super_percentile)
    counter = n_sessions * (100 - super_percentile) / 100
    reward_threshold = quantile(rewards_batch, super_percentile / 100)
    super_states = zeros(Float32, 0, size(states_batch, 2), size(states_batch, 3))
    super_actions = Matrix{Int}(undef, 0, size(actions_batch, 2))
    super_rewards = Vector{Float32}(undef, 0)

    for i in 1:size(states_batch, 1)
        if rewards_batch[i] >= reward_threshold - 0.000001 && counter > 0
            super_states = append_2d_to_3d(super_states, states_batch[i, :, :])
            super_actions = vcat(super_actions, reshape(actions_batch[i, :], 1, size(actions_batch, 2)))
            push!(super_rewards, rewards_batch[i])
        end
        counter -= 1
    end

    return super_states, super_actions, super_rewards
end


######### DEFINING THE MODEL AND TRAINING FUCNTION #########

function train(board_size, write_all, write_best, filename, slow)

    n = board_size

    update_n(board_size) # Updates a global variable in reward_function.jl file

    if slow
        dist_matrix = generate_distance_matrix(1)
    else
        dist_matrix = generate_distance_matrix(n)
    end

    len_word = n^2  # Length of the word to generate
    observation_space = 2 * len_word
    len_game = len_word

    first_layer_neurons = 128
    second_layer_neurons = 64
    third_layer_neurons = 4

    # Define the neural network architecture (similar to PyTorch)
    model = Chain(
        Dense(observation_space, first_layer_neurons, relu),
        Dense(first_layer_neurons, second_layer_neurons, relu),
        Dense(second_layer_neurons, third_layer_neurons, relu),
        Dense(third_layer_neurons, 1, σ)
    )

    # Create an instance of the neural network
    net = model

    # Defining the loss function and optimizer (similar to PyTorch)
    criterion(y_pred, y_true) = Flux.binarycrossentropy(y_pred, y_true)
    optimizer = Optimise.ADAM(learning_rate)

    # Define variables (use Float32 for tensors)
    global super_states = Array{Float32}(undef, 0, len_game, observation_space)
    global super_actions = Int[]
    global super_rewards = Float32[]

    sessgen_time = 0.0
    fit_time = 0.0
    score_time = 0.0

    pass_threshold =  0.25*n # the mean_all_reward must be greater than this threshold to pass the test
    counter = 0

    for i in 1:20000
        println("\n GENERATION $i")
        tic = time()
        sessions = generate_session(net, n_sessions, n, dist_matrix, len_word, len_game, observation_space, 1, slow) # change 0 to 1 to print out how much time each step in generate_session takes
        sessgen_time = time() - tic
    
        tic = time()
    
        states_batch = sessions[1]  # Assuming generate_session returns the states in sessions[1]
        actions_batch = sessions[2]  # Assuming generate_session returns the actions in sessions[2]
        rewards_batch = sessions[3]  # Assuming generate_session returns the rewards in sessions[3]
        states_batch = permutedims(states_batch, (1, 3, 2))  # Transpose the tensor
        
        states_batch = vcat(states_batch, super_states)

        if i > 1
            actions_batch = vcat(actions_batch, super_actions)  
        end
    
        rewards_batch = vcat(rewards_batch, super_rewards)  
    
        randomcomp_time = time() - tic
        tic = time()
    
        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile) # pick the sessions to learn from
        select1_time = time() - tic
    
        tic = time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, super_percentile) # pick the sessions to survive
        select2_time = time() - tic
    
        tic = time()
        super_sessions = sort_by_last_vector(super_sessions[1], super_sessions[2], super_sessions[3])
        select3_time = time() - tic

        tic = time()
        # Backward pass (gradients calculation) and optimization (similar to PyTorch)
        outputs = zeros(Float32, size(elite_states, 1), 1)
        for i in 1:size(elite_states,1)
            outputs[i] = model(elite_states[i, :, :])[1]
        end
        loss = criterion(outputs, elite_actions)
        grads = gradient(() -> loss, params(model))
        Optimise.update!(optimizer, params(model), grads)
        fit_time = time()-tic

        tic = time()

        for i in 1:size(super_sessions[1], 1)
            super_states = append_2d_to_3d(super_states, super_sessions[1][i, :, :])
            if i == 1
                super_actions = reshape(super_sessions[2][i, :], 1, size(super_sessions[2], 2))
            else
                super_actions = vcat(super_actions, reshape(super_sessions[2][i, :], 1, size(super_sessions[2], 2)))
            end
            push!(super_rewards, super_sessions[3][i])
        end

        sort!(rewards_batch)
        mean_all_reward = mean(rewards_batch[end-200:end])
        mean_best_reward = mean(super_rewards)

        if mean_all_reward < 1
            counter+=1
        end

        score_time = time() - tic

        println("\n$i. Best individuals: ", reverse(sort(super_rewards)))

        # Uncomment the line below to print out how much time each step in this loop takes.
        println("Mean reward: $mean_all_reward\nSessgen: $sessgen_time, other: $randomcomp_time, select1: $select1_time, select2: $select2_time, select3: $select3_time, fit: $fit_time, score: $score_time")

        # Uncomment the line below to print out the mean best reward
        println("Mean best reward: $mean_best_reward")

            # Make a new folder if 'Data' folder does not exist
        if !isdir("Data")
            mkdir("Data")
        end

        if write_all
            if i % 20 == 1  # Write all important info to files every 20 iterations
                # Writing super_actions to a binary file
                open(joinpath("Data", "$(filename)_best_species_pickle.txt"), "w") do fp
                    write(fp, super_actions)
                end
                
                # Writing super_actions to a text file
                open(joinpath("Data", "$(filename)_best_species.txt"), "w") do f
                    for item in eachrow(super_actions)
                        write(f, string(convert_to_board(item, n), "\n"))
                    end
                end
                
                # Writing super_rewards to a text file
                open(joinpath("Data", "$(filename)_best_species_rewards.txt"), "w") do f
                    for item in super_rewards
                        write(f, string(item, "\n"))
                    end
                end
                
                # Appending mean_all_reward to a text file
                open(joinpath("Data", "$(filename)_best_100_rewards.txt"), "a") do f
                    write(f, string(mean_all_reward, "\n"))
                end
                
                # Appending mean_best_reward to a text file
                open(joinpath("Data", "$(filename)_best_elite_rewards.txt"), "a") do f
                    write(f, string(mean_best_reward, "\n"))
                end
                
                if i % 200 == 2  # To create a timeline, like in Figure 3
                    open(joinpath("Data", "$(filename)_best_species_timeline.txt"), "a") do f
                        write(f, string(convert_to_board(super_actions[1], n), "\n"))
                    end
                end
            end
        end

        if write_best && mean_best_reward > pass_threshold
            # Writing super_actions to a binary file
            open(joinpath("Data", "$(filename)_best_species_pickle.txt"), "w") do fp
                write(fp, super_actions)
            end
            
            # Writing super_actions to a text file
            open(joinpath("Data", "$(filename)_best_species.txt"), "w") do f
                for item in eachrow(super_actions)
                    write(f, string(convert_to_board(item, n), "\n"))
                end
            end
            
            # Writing super_rewards to a text file
            open(joinpath("Data", "$(filename)_best_species_rewards.txt"), "w") do f
                for item in super_rewards
                    write(f, string(item, "\n"))
                end
            end
            
            # Appending mean_all_reward to a text file
            open(joinpath("Data", "$(filename)_best_100_rewards.txt"), "a") do f
                write(f, string(mean_all_reward, "\n"))
            end
            
            # Appending mean_best_reward to a text file
            open(joinpath("Data", "$(filename)_best_elite_rewards.txt"), "a") do f
                write(f, string(mean_best_reward, "\n"))
            end
            
            # Writing super_actions[1] to a timeline text file
            open(joinpath("Data", "$(filename)_best_species_timeline.txt"), "a") do f
                write(f, string(convert_to_board(super_actions[1,:], n), "\n"))
            end
        end

        if counter > 500 && mean_best_reward > pass_threshold
            # Writing super_actions to a binary file
            open(joinpath("Data", "$(filename)_best_species_pickle.txt"), "w") do fp
                write(fp, super_actions)
            end
            
            # Writing super_actions to a text file
            open(joinpath("Data", "$(filename)_best_species.txt"), "w") do f
                for item in eachrow(super_actions)
                    write(f, string(convert_to_board(item, n), "\n"))
                end
            end
            
            # Writing super_rewards to a text file
            open(joinpath("Data", "$(filename)_best_species_rewards.txt"), "w") do f
                for item in super_rewards
                    write(f, string(item, "\n"))
                end
            end
            
            # Appending mean_all_reward to a text file
            open(joinpath("Data", "$(filename)_best_100_rewards.txt"), "a") do f
                write(f, string(mean_all_reward, "\n"))
            end
            
            # Appending mean_best_reward to a text file
            open(joinpath("Data", "$(filename)_best_elite_rewards.txt"), "a") do f
                write(f, string(mean_best_reward, "\n"))
            end
            
            # Writing super_actions[1] to a timeline text file
            open(joinpath("Data", "$(filename)_best_species_timeline.txt"), "a") do f
                write(f, string(convert_to_board(super_actions[1,:], n), "\n"))
            end
            
            return 
        end
    end

end
