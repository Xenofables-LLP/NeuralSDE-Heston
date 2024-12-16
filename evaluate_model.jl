using Flux
using Zygote
using BSON
using Plots
using Statistics
using Random: randn
using Dates

# Helper function for logging
function log(message::String)
    Zygote.ignore() do
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss")
        println("[$timestamp] $message")
    end
end

log("Starting testing script for Neural SDE...")

# Load the trained model parameters
bson_file_path = "neural_sde_model.bson"
log("Loading model parameters from '$bson_file_path'...")
bson_data = BSON.load(bson_file_path)
if haskey(bson_data, :θ_vec)
    θ_vec = bson_data[:θ_vec]
    log("Model parameters loaded successfully!")
else
    error("Key 'θ_vec' not found in the BSON file. Available keys: $(keys(bson_data))")
end

# Ensure the same layer configurations
drift_layers = [(10, 32), (32, 32), (32, 2)]
diffusion_layers = [(10, 32), (32, 32), (32, 2)]

# Define functions to rebuild the model (reused from training script)
function calculate_total_params(layers)
    total_params = 0
    for (in_dim, out_dim) in layers
        total_params += in_dim * out_dim + out_dim
    end
    return total_params
end

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
    l1, idx = build_dense(θ_val, idx, 10, 32, leakyrelu)
    l2, idx = build_dense(θ_val, idx, 32, 32, leakyrelu)
    l3, idx = build_dense(θ_val, idx, 32, 2, identity)
    Chain(l1, l2, l3)
end

function build_models(θ)
    drift_param_count = calculate_total_params(drift_layers)
    diff_param_count = calculate_total_params(diffusion_layers)

    drift_params = θ[1:drift_param_count]
    diff_params = θ[drift_param_count+1:end]

    nn_drift_built = build_chain(drift_params)
    nn_diffusion_built = build_chain(diff_params)
    return nn_drift_built, nn_diffusion_built
end

# Ensure the preprocess_data function matches training script
function preprocess_data(dataset, train_ratio=0.8)
    log("Dataset column names: $(names(dataset))")
    log("Dataset CLOSE stats: mean = $(mean(dataset.CLOSE)), std = $(std(dataset.CLOSE))")

    # Extract features and normalize
    CLOSE = Float32.(dataset.CLOSE)
    delta = Float32.(dataset.delta)
    gamma = Float32.(dataset.gamma)
    theta = Float32.(dataset.theta)
    vega = Float32.(dataset.vega)
    rho = Float32.(dataset.rho)
    bid = Float32.(dataset.bid)
    ask = Float32.(dataset.ask)
    strike_normalized = Float32.(dataset.strike ./ dataset.CLOSE)
    call_put_binary = Float32.((dataset.call_put .== "Call") .* 1 .+ (dataset.call_put .!= "Call") .* -1)

    midpoint = Float32.((bid + ask) ./ 2)
    T = Float32.(Dates.value.(dataset.expiration .- dataset.date)) ./ 365.0

    # Normalize function with clipping for stability
    function normalize(data, mean_val, std_val; clip_min=-3.0, clip_max=3.0)
        normalized = (data .- mean_val) ./ std_val
        return clamp.(normalized, clip_min, clip_max)
    end

    # Compute normalization statistics for individual features into a 2d array
    features = hcat(
        CLOSE,
        delta,
        gamma,
        theta,
        vega,
        rho,
        T,
        strike_normalized,
        midpoint,
        call_put_binary
    )

    # Compute mean and standard deviation
    feature_mean = mean(features, dims=1)
    feature_std = std(features, dims=1)

    # Normalize features
    normalized_features = (features .- feature_mean) ./ feature_std
    normalized_features = clamp.(normalized_features, -3.0f0, 3.0f0)  # Apply clipping

    # Split into train and test sets
    num_train = Int(size(dataset, 1) * train_ratio)
    x_train = normalized_features[1:num_train, :]
    y_train = normalized_features[1:num_train, 1:2]  # Target is CLOSE and delta

    x_test = normalized_features[num_train+1:end, :]
    y_test = normalized_features[num_train+1:end, 1:2]
    
    # Convert to Float32 for consistency
    x_train = Float32.(x_train)
    y_train = Float32.(y_train)
    x_test = Float32.(x_test)
    y_test = Float32.(y_test)

    log("x_train stats: mean = $(mean(x_train, dims=1)), std = $(std(x_train, dims=1))")
    log("y_train stats: mean = $(mean(y_train, dims=1)), std = $(std(y_train, dims=1))")

    return x_train, y_train, x_test, y_test, feature_mean, feature_std
end

# Load testing dataset (ensure preprocessing matches training script)
log("Loading testing dataset and preprocessing...")
x_train, y_train, x_test, y_test, feature_mean, feature_std = preprocess_data(dataset)
log("Testing dataset loaded and preprocessed!")

# Define forward pass for testing (reused from training script)
function forward_batch(x_batch, θ; dt=1.0f0 / size(x_batch, 1))
    nn_drift_built, nn_diffusion_built = build_models(θ)
    num_samples = size(x_batch, 1)

    dW1_values = sqrt(dt) .* randn(Float32, num_samples)
    dZ_values = sqrt(dt) .* randn(Float32, num_samples)
    dW2_values = -0.7f0 .* dW1_values .+ sqrt(1 - (-0.7f0)^2) .* dZ_values

    S_values = map(i -> begin
        f = nn_drift_built(x_batch[i, :])
        g = nn_diffusion_built(x_batch[i, :])
        fS, fV = f[1], f[2]
        gS, gV = g[1], g[2]
        V_next = max(0.0f0, fV * dt + gV * dW2_values[i])
        S_next = fS * dt + sqrt(max(0.0f0, V_next)) * gS * dW1_values[i]
        (S_next, V_next)
    end, 1:num_samples)

    return map(first, S_values), map(last, S_values)
end

# Perform predictions on test data
log("Running forward pass on test data...")
S_pred, V_pred = forward_batch(x_test, θ_vec)
log("Predictions completed!")

# Evaluate performance
log("Calculating Mean Squared Errors (MSE) for predictions...")
mse_S = mean((S_pred .- y_test[:, 1]).^2)
mse_V = mean((V_pred .- y_test[:, 2]).^2)
log("MSE for S (CLOSE): $mse_S")
log("MSE for V (delta): $mse_V")

# Plot predictions vs. actual values
log("Plotting predictions vs actual values...")
p1 = scatter(1:length(S_pred), S_pred, label="Predicted CLOSE", xlabel="Sample Index", ylabel="Value", title="Predicted vs Actual CLOSE")
scatter!(1:length(S_pred), y_test[:, 1], label="Actual CLOSE")

p2 = scatter(1:length(V_pred), V_pred, label="Predicted delta", xlabel="Sample Index", ylabel="Value", title="Predicted vs Actual delta")
scatter!(1:length(V_pred), y_test[:, 2], label="Actual delta")

# Save the plots
log("Saving plots as images...")
savefig(p1, "predicted_vs_actual_close.png")
savefig(p2, "predicted_vs_actual_delta.png")
log("Plots saved successfully!")

log("Displaying plots...")
plot(p1)
plot(p2)

log("Testing completed!")
