using Flux
using ReverseDiff
using BSON
using CSV
using DataFrames
using Statistics
using Random: randn

println("Starting Functional Neural SDE with ReverseDiff (no Float32 casts during forward pass)...")

# Load and Preprocess Dataset
file_path = "options-A-merged.csv"
println("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
println("Dataset loaded successfully!")

num_rows, num_cols = size(dataset)
println("Rows and columns in dataset: $num_rows, $num_cols")

function preprocess_data(dataset, train_ratio=0.8)
    CLOSE = Float32.(dataset.CLOSE)
    delta = Float32.(dataset.delta)

    num_train = Int(size(dataset, 1) * train_ratio)
    S_train = CLOSE[1:num_train]
    V_train = delta[1:num_train]

    max_S = maximum(S_train)
    max_V = maximum(V_train)
    if max_S == 0.0f0 || max_V == 0.0f0
        error("Max in 'CLOSE' or 'delta' is zero, normalization not possible.")
    end

    S_train_scaled = S_train ./ max_S
    V_train_scaled = V_train ./ max_V
    return S_train_scaled, V_train_scaled, max_S, max_V
end

S_train_scaled, V_train_scaled, max_S, max_V = preprocess_data(dataset)
u0 = [S_train_scaled[1], V_train_scaled[1]]  # Both Float32

num_train_samples = length(S_train_scaled)
t_train = collect(Float32, 0.0f0:1.0f0/(num_train_samples - 1):1.0f0)
dt = t_train[2] - t_train[1]

# Neural network architecture parameters
# Each network: Dense(2->16, relu), Dense(16->16, relu), Dense(16->2)
# Params count per network: 354
# For two networks (drift & diffusion): 708 params total
const DRIFT_PARAMS = 354
const TOTAL_PARAMS = 708

# Fixed correlation parameter
ρ = -0.7f0

# Initialize parameters θ_vec
θ_vec = Float32.(randn(TOTAL_PARAMS)*0.01)

# Helper to build Dense layers from a parameter vector
function build_dense(θ_val, idx, in_dim, out_dim, σ=relu)
    W_size = out_dim * in_dim
    W = reshape(θ_val[idx:idx+W_size-1], out_dim, in_dim)
    idx += W_size
    b = θ_val[idx:idx+out_dim-1]
    idx += out_dim
    return Dense(W, b, σ), idx
end

function build_chain(θ_val)
    idx = 1
    l1, idx = build_dense(θ_val, idx, 2, 16, relu)
    l2, idx = build_dense(θ_val, idx, 16, 16, relu)
    l3, idx = build_dense(θ_val, idx, 16, 2, identity)
    Chain(l1, l2, l3)
end

function build_models(θ)
    # Extract the raw values from θ (if tracked, this creates a dependency ReverseDiff can follow)
    θ_val = ReverseDiff.value.(θ)

    drift_params = θ_val[1:DRIFT_PARAMS]
    diff_params = θ_val[DRIFT_PARAMS+1:end]

    nn_drift_built = build_chain(drift_params)
    nn_diffusion_built = build_chain(diff_params)
    return nn_drift_built, nn_diffusion_built
end

function forward_sde(u0, θ; dt=dt)
    nn_drift_built, nn_diffusion_built = build_models(θ)

    # Store states in Vector{Any} so we can insert tracked values
    S_values = Any[u0[1]]
    V_values = Any[u0[2]]

    for i in 2:length(S_train_scaled)
        S_prev = S_values[end]
        V_prev = V_values[end]
        
        # Just use the values as is. They are Float32 or tracked Float32, no need to cast.
        state_input = [S_prev, V_prev]  # Vector of length 2

        f = nn_drift_built(state_input)
        fS, fV = f[1], f[2]

        g = nn_diffusion_built(state_input)
        gS, gV = g[1], g[2]

        dW1 = sqrt(dt)*randn(Float32)
        dZ  = sqrt(dt)*randn(Float32)
        dW2 = ρ * dW1 + sqrt(1 - ρ^2)*dZ

        S_next = S_prev + fS*dt + gS*dW1
        V_next = V_prev + fV*dt + gV*dW2

        # Append next states
        push!(S_values, S_next)
        push!(V_values, V_next)
    end
    return S_values, V_values
end

function loss(θ)
    S_pred, V_pred = forward_sde(u0, θ)
    # S_pred, V_pred may contain tracked values. Arithmetic with Float32 arrays is fine.
    # mean and arithmetic should be differentiable by ReverseDiff.
    mse_S = mean((S_pred .- S_train_scaled).^2)
    mse_V = mean((V_pred .- V_train_scaled).^2)
    mse_S + mse_V
end

function train(epochs, lr, θ_vec)
    println("Starting training process...")
    losses = Float32[]

    for epoch in 1:epochs
        grad_θ = ReverseDiff.gradient(loss, θ_vec)
        θ_vec = θ_vec .- lr .* grad_θ

        current_loss = loss(θ_vec)
        push!(losses, current_loss)
        println("Epoch $epoch completed. Loss: $current_loss")
    end

    BSON.@save "neural_sde_model.bson" θ_vec ρ
    println("Model parameters saved to neural_sde_model.bson")
    return θ_vec, losses
end

# Example training run
lr = 0.001f0
θ_vec, losses = train(10, lr, θ_vec)

function load_model(save_path="neural_sde_model.bson")
    println("Loading model from $save_path...")
    model_data = BSON.load(save_path)
    θ_loaded = model_data[:θ_vec]
    ρ_loaded = model_data[:ρ]
    println("Model loaded successfully!")
    return θ_loaded, ρ_loaded
end

θ_loaded, ρ_loaded = load_model()

println("Testing loaded model with final parameters...")
S_pred, V_pred = forward_sde(u0, θ_loaded)
@show mean((S_pred .- S_train_scaled).^2)
@show mean((V_pred .- V_train_scaled).^2)