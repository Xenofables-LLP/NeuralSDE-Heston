using DifferentialEquations
using Flux
using BSON
using CSV
using DataFrames
using Plots  # For plotting the results

println("Starting Testing Heston Model Neural SDE...")

# Load Dataset (for testing)
file_path = "options-A-merged.csv"
println("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
println("Dataset loaded successfully!")

# Preprocess the data similarly to the training data
function preprocess_data(dataset)
    CLOSE = Float32.(dataset.CLOSE)  # Convert `CLOSE` column to Float32
    delta = Float32.(dataset.delta)  # Convert `delta` column to Float32

    println("Length of CLOSE: ", length(CLOSE))
    println("Length of delta: ", length(delta))

    # Normalize data
    max_S = maximum(CLOSE)
    max_V = maximum(delta)

    S_scaled = CLOSE ./ max_S
    V_scaled = delta ./ max_V

    return S_scaled, V_scaled
end

S_scaled, V_scaled = preprocess_data(dataset)

# Load the trained model
function load_model(save_path="heston_model.bson")
    println("Loading model from $save_path...")
    model_data = BSON.load(save_path)
    nn_drift_loaded = model_data[:nn_drift]
    θ_loaded = model_data[:θ]
    println("Model loaded successfully!")
    return nn_drift_loaded, θ_loaded
end

nn_drift_loaded, θ_loaded = load_model()

# Define Drift and Diffusion Functions for testing
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

# Test Model
function test_model(nn_drift, θ, S_scaled, V_scaled, tspan=(0.0f0, 1.0f0))
    u0 = [S_scaled[1], V_scaled[1]]
    prob = SDEProblem(drift!, diffusion!, u0, tspan, θ)

    # Solve the problem
    sol = solve(prob, EM(), sensealg=QuadratureAdjoint(), dt=0.01)

    # Interpolating solution to match testing data
    t_train = collect(range(tspan[1], tspan[2], length=length(S_scaled)))
    S_pred = [sol(t)[1] for t in t_train]
    V_pred = [sol(t)[2] for t in t_train]

    # Mean Squared Error calculation
    mse_S = mean((S_pred .- S_scaled).^2)
    mse_V = mean((V_pred .- V_scaled).^2)

    println("Mean Squared Error for Stock Prices: $mse_S")
    println("Mean Squared Error for Volatilities: $mse_V")

    # Plotting the results
    p1 = plot(t_train, S_pred, label="Predicted Stock Prices", xlabel="Time", ylabel="Stock Price", title="Stock Price Prediction")
    plot!(p1, t_train, S_scaled, label="Actual Stock Prices", linestyle=:dash)

    p2 = plot(t_train, V_pred, label="Predicted Volatility", xlabel="Time", ylabel="Volatility", title="Volatility Prediction")
    plot!(p2, t_train, V_scaled, label="Actual Volatility", linestyle=:dash)

    display(p1)
    display(p2)
end

# Run test
test_model(nn_drift_loaded, θ_loaded, S_scaled, V_scaled)

println("Testing complete.")
