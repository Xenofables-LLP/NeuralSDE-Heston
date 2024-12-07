using Flux
using Zygote
using Optimisers
using DifferentialEquations
using DiffEqFlux
using BSON
using CSV
using DataFrames
using Statistics

println("Starting Heston Model Neural SDE Implementation...")

# Load Dataset
file_path = "options-A-merged-Small.csv"
println("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
println("Dataset loaded successfully!")

num_rows, num_cols = size(dataset)
println("Rows and columns in dataset: $num_rows, $num_cols")

# Preprocessing Function
function preprocess_data(dataset, train_ratio=0.8)
    println("Preprocessing data...")

    CLOSE = Float32.(dataset.CLOSE)
    delta = Float32.(dataset.delta)

    println("Length of CLOSE: ", length(CLOSE))
    println("Length of delta: ", length(delta))

    num_train = Int(size(dataset, 1) * train_ratio)
    println("Number of training samples: ", num_train)

    S_train = CLOSE[1:num_train]
    V_train = delta[1:num_train]

    max_S = maximum(S_train)
    max_V = maximum(V_train)

    if max_S == 0.0 || max_V == 0.0
        error("Maximum value in 'CLOSE' or 'delta' column is zero, normalization not possible.")
    end

    S_train_scaled = S_train ./ max_S
    V_train_scaled = V_train ./ max_V

    println("Data normalization complete!")
    return S_train_scaled, V_train_scaled
end

S_train_scaled, V_train_scaled = preprocess_data(dataset)

u0 = [S_train_scaled[1], V_train_scaled[1]]
tspan = (0.0f0, 1.0f0)

# Drift and Diffusion Functions
function drift!(du, u, p, t)
    S, V = u
    α, β, γ = p[1], p[2], p[3]
    du[1] = α * S
    du[2] = β * (γ - V)
end

function diffusion!(du, u, p, t)
    S, V = u
    σ, ρ = p[4], p[5]
    du[1] = σ * S * √V
    du[2] = ρ * √V
end

# Neural Network for Parameterizing Drift and Diffusion
nn_drift = Flux.Chain(
    Flux.Dense(2, 16, Flux.relu),
    Flux.Dense(16, 16, Flux.relu),
    Flux.Dense(16, 2)
)

θ = [0.1f0, 0.5f0, 0.1f0, 0.2f0, 0.3f0]

# Define the SDE Problem
prob = SDEProblem(drift!, diffusion!, u0, tspan, θ)

# Loss Function
function loss(θ)
    println("Computing loss with θ = ", θ)
    
    # Remake the problem with updated parameters
    prob_p = remake(prob, p=θ)
    
    # Solve the SDE using an appropriate solver
    num_train_samples = length(S_train_scaled)
    t_train = collect(Float32, 0.0:1.0/(num_train_samples - 1):1.0)
    sol = solve(prob_p, EM(), dt=0.01, saveat=t_train)
    
    # Extract predictions
    S_pred_interp = Array(sol[1, :])
    V_pred_interp = Array(sol[2, :])
    
    # Compute mean squared error
    mse_S = mean((S_pred_interp .- S_train_scaled).^2)
    mse_V = mean((V_pred_interp .- V_train_scaled).^2)
    
    println("MSE_S: ", mse_S, ", MSE_V: ", mse_V)
    return mse_S + mse_V
end

# Training Function
function train(prob, epochs, opt, θ, save_path="heston_model.bson")
    println("Starting training process...")
    losses = []

    for epoch in 1:epochs
        println("Epoch $epoch...")
        try
            # Wrap θ in a compatible format for Zygote and Optimisers
            θ_params = Flux.Params([θ])

            # Compute the gradient of the loss function w.r.t. θ
            grads = Zygote.gradient(() -> loss(θ), θ_params)

            # Update parameters using the optimizer
            for (param, grad) in zip(θ_params, grads)
                Optimisers.update!(opt, param, grad)
            end

            # Calculate current loss
            current_loss = loss(θ)
            push!(losses, current_loss)
            println("Epoch $epoch completed. Loss: $current_loss")
        catch e
            println("Error during training at epoch $epoch: $e")
        end
    end

    BSON.@save save_path nn_drift θ
    println("Model saved to $save_path")
end

opt = Optimisers.Adam(0.001)
train(prob, 10, opt, θ)

# Load Function
function load_model(save_path="heston_model.bson")
    println("Loading model from $save_path...")
    model_data = BSON.load(save_path)
    nn_drift_loaded = model_data[:nn_drift]
    θ_loaded = model_data[:θ]
    println("Model loaded successfully!")
    return nn_drift_loaded, θ_loaded
end

nn_drift_loaded, θ_loaded = load_model()

println("Testing loaded model...")
@show nn_drift_loaded
@show θ_loaded
