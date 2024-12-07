using Flux
using ReverseDiff
using BSON
using CSV
using DataFrames
using Statistics
using Random: randn
using Plots

# -----------------------------------------
# Load the Model and Parameters
# -----------------------------------------
function load_model(path="neural_sde_model.bson")
    println("Loading model from $path...")
    model_data = BSON.load(path)
    θ_loaded = model_data[:θ_vec]
    ρ_loaded = model_data[:ρ]
    max_S = model_data[:max_S]
    max_V = model_data[:max_V]
    println("Model loaded successfully!")
    return θ_loaded, ρ_loaded, max_S, max_V
end

θ_loaded, ρ_loaded, max_S, max_V = load_model("neural_sde_model.bson")

# -----------------------------------------
# Load and Preprocess Data
# -----------------------------------------
file_path = "options-A-merged.csv"
println("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
println("Dataset loaded successfully!")

# Preprocessing function must match what was used during training
function preprocess_data(dataset; train_ratio=0.8)
    CLOSE = Float32.(dataset.CLOSE)
    delta = Float32.(dataset.delta)

    num_train = Int(floor(size(dataset, 1) * train_ratio))
    S_train = CLOSE[1:num_train]
    V_train = delta[1:num_train]

    S_test = CLOSE[num_train+1:end]
    V_test = delta[num_train+1:end]

    # Use the same max_S and max_V from training (loaded from the model file)
    if max_S == 0f0 || max_V == 0f0
        error("Max values are zero, cannot normalize.")
    end

    S_train_scaled = S_train ./ max_S
    V_train_scaled = V_train ./ max_V
    S_test_scaled = S_test ./ max_S
    V_test_scaled = V_test ./ max_V

    return S_train_scaled, V_train_scaled, S_test_scaled, V_test_scaled
end

S_train_scaled, V_train_scaled, S_test_scaled, V_test_scaled = preprocess_data(dataset)

num_train_samples = length(S_train_scaled)
num_test_samples = length(S_test_scaled)
println("Training samples: $num_train_samples, Test samples: $num_test_samples")

# Define time step and steps (must match training conditions)
# Suppose during training you used t in [0,1] with num_train_samples points
t_train = range(0f0, 1f0, length=num_train_samples)
dt = t_train[2] - t_train[1]

# -----------------------------------------
# Model Architecture Setup
# -----------------------------------------
# These constants must match training code
const DRIFT_PARAMS = 354
const TOTAL_PARAMS = 708

function build_dense(θ_val, idx, in_dim, out_dim, σ)
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
    drift_params = θ[1:DRIFT_PARAMS]
    diff_params = θ[DRIFT_PARAMS+1:end]

    nn_drift_built = build_chain(drift_params)
    nn_diffusion_built = build_chain(diff_params)
    return nn_drift_built, nn_diffusion_built
end

function forward_sde(u0, θ; dt=dt, steps::Int)
    nn_drift_built, nn_diffusion_built = build_models(θ)

    S_values = Any[u0[1]]
    V_values = Any[u0[2]]

    for i in 2:steps
        S_prev = S_values[end]
        V_prev = V_values[end]

        state_input = [S_prev, V_prev]

        f = nn_drift_built(state_input)
        fS, fV = f[1], f[2]

        g = nn_diffusion_built(state_input)
        gS, gV = g[1], g[2]

        dW1 = sqrt(dt)*randn(Float32)
        dZ  = sqrt(dt)*randn(Float32)
        dW2 = ρ_loaded * dW1 + sqrt(1 - ρ_loaded^2)*dZ

        S_next = S_prev + fS*dt + gS*dW1
        V_next = V_prev + fV*dt + gV*dW2

        push!(S_values, S_next)
        push!(V_values, V_next)
    end
    return S_values, V_values
end

# -----------------------------------------
# Testing the Model
# -----------------------------------------
if num_test_samples > 1
    # Initial condition from test set
    u0_test = [S_test_scaled[1], V_test_scaled[1]]

    # Forward simulate on test set
    S_pred_test, V_pred_test = forward_sde(u0_test, θ_loaded, steps=num_test_samples)

    # Compute MSE
    mse_S_test = mean((S_pred_test .- S_test_scaled).^2)
    mse_V_test = mean((V_pred_test .- V_test_scaled).^2)
    println("Test MSE on S: $mse_S_test")
    println("Test MSE on V: $mse_V_test")

    # Plot predictions vs actual
    p1 = plot(1:num_test_samples, S_test_scaled, label="S Actual (Test)", title="S Prediction vs Actual (Test)", xlabel="Time Index", ylabel="S (scaled)")
    plot!(p1, 1:num_test_samples, S_pred_test, label="S Predicted (Test)")

    p2 = plot(1:num_test_samples, V_test_scaled, label="V Actual (Test)", title="V Prediction vs Actual (Test)", xlabel="Time Index", ylabel="V (scaled)")
    plot!(p2, 1:num_test_samples, V_pred_test, label="V Predicted (Test)")

    display(p1)
    display(p2)
else
    println("Not enough test data to perform testing.")
end
