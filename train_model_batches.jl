using Flux
using Zygote
using BSON
using CSV
using DataFrames
using Statistics
using Random: randn
using Plots
using LinearAlgebra
using Dates

# Helper function for logging with timestamps
function log(message::String)
    Zygote.ignore() do
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss")
        println("[$timestamp] $message")
    end
end

log("Starting Functional Neural SDE with Zygote...")

# Load and Preprocess Dataset
file_path = "options-A-merged.csv"
log("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
log("Dataset loaded successfully!")

num_rows, num_cols = size(dataset)
log("Rows and columns in dataset: $num_rows, $num_cols")

# Preprocess the dataset
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

x_train, y_train, x_test, y_test, feature_mean, feature_std = preprocess_data(dataset)
log("Initial feature matrix size: $(size(x_train))")
log("Initial target matrix size: $(size(y_train))")

# Xavier initialization
function xavier_initialization(drift_layers, diffusion_layers)
    function init_layers(layers)
        θ_vals = []
        for (in_dim, out_dim) in layers
            W = Float32.(randn(out_dim, in_dim) .* sqrt(2.0 / (in_dim + out_dim)))
            b = Float32.(zeros(out_dim))
            push!(θ_vals, vec(W), b)
        end
        return reduce(vcat, θ_vals)
    end

    drift_params = init_layers(drift_layers)
    diffusion_params = init_layers(diffusion_layers)
    return vcat(drift_params, diffusion_params)
end

# Layer configurations
drift_layers = [(10, 32), (32, 32), (32, 2)]
diffusion_layers = [(10, 32), (32, 32), (32, 2)]

const TOTAL_PARAMS = sum([in_dim * out_dim + out_dim for (in_dim, out_dim) in drift_layers]) +
                     sum([in_dim * out_dim + out_dim for (in_dim, out_dim) in diffusion_layers])
log("Recalculated TOTAL_PARAMS: $TOTAL_PARAMS")

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

# Forward pass for one batch
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

# Training configuration
batch_size = 128
epochs = 10
lr = 0.01f0
θ_vec = xavier_initialization(drift_layers, diffusion_layers)
# Adjust loss weights dynamically based on magnitude
function batch_loss(x_batch, y_batch, θ, batch_idx)
    S_pred, V_pred = forward_batch(x_batch, θ)
    mse_S = mean((S_pred .- y_batch[:, 1]).^2)
    mse_V = mean((V_pred .- y_batch[:, 2]).^2)

    weight_S = 1.0f0 / (1.0f0 + mse_S)
    weight_V = 1.0f0 / (1.0f0 + mse_V)

    # Weighted loss
    loss = weight_S * mse_S + weight_V * mse_V

    # Debugging
    # if batch_idx % 50 == 0
    #     log("Debugging Batch $batch_idx:")
    #     log("mse_S: $mse_S, mse_V: $mse_V, weight_S: $weight_S, weight_V: $weight_V")
    #     log("S_pred (first 5): $(S_pred[1:5])")
    #     log("V_pred (first 5): $(V_pred[1:5])")
    # end

    return loss
end

# Training with batch processing
function train_model_batch_with_logging(x_train, y_train, x_test, y_test, θ_vec, epochs, batch_size, lr)
    log("Starting batch training...")
    num_samples = size(x_train, 1)
    num_batches = div(num_samples, batch_size)

    # Initialize loss history
    train_loss_history = []
    val_loss_history = []

    for epoch in 1:epochs
        log("Epoch $epoch...")
        epoch_loss = 0.0f0  # To accumulate loss for the epoch

        for batch_idx in 0:(num_batches-1)
            start_idx = batch_idx * batch_size + 1
            end_idx = min((batch_idx + 1) * batch_size, num_samples)
            x_batch = x_train[start_idx:end_idx, :]
            y_batch = y_train[start_idx:end_idx, :]

            # Compute loss and gradient for this batch
            function compute_loss(θ)
                return batch_loss(x_batch, y_batch, θ, batch_idx) + 1e-4 * sum(θ.^2)
            end

            loss_value, pullback_fn = Zygote.pullback(compute_loss, θ_vec)  # Get the loss and pullback
            grad = pullback_fn(1)[1]  # Compute gradients
            epoch_loss += loss_value  # Accumulate batch loss

            # Apply gradient clipping during parameter updates
            clipped_grad = clamp.(grad, -10.0f0, 10.0f0)
            θ_vec .= θ_vec .- lr .* clipped_grad

            if any(isnan.(θ_vec)) || any(isinf.(θ_vec))
                error("θ_vec contains NaN or Inf values!")
            end

            # Log batch loss
            log("Batch $batch_idx: Loss = $loss_value")
        end

        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        push!(train_loss_history, avg_epoch_loss)
        log("Epoch $epoch completed. Average Training Loss: $avg_epoch_loss")

        # Calculate validation loss
        val_loss = calculate_validation_loss(x_test, y_test, θ_vec)
        push!(val_loss_history, val_loss)
        log("Epoch $epoch Validation Loss: $val_loss")
    end

    # Save model parameters explicitly
    log("Saving model to: $(pwd())/neural_sde_model.bson")
    log("Saving θ_vec with size: $(length(θ_vec))")
    BSON.@save "neural_sde_model.bson" θ_vec=θ_vec train_loss_history=train_loss_history val_loss_history=val_loss_history
    log("Model saved successfully.")

    return train_loss_history, val_loss_history
end

function calculate_validation_loss(x_test, y_test, θ)
    val_samples = size(x_test, 1)
    val_loss = 0.0f0

    for i in 1:val_samples
        x_sample = reshape(x_test[i, :], 1, :)  # Reshape to 2D array
        y_sample = y_test[i, :]
        function compute_loss(θ)
            S_pred, V_pred = forward_batch(x_sample, θ)
            mse_S = mean((S_pred .- y_sample[1]).^2)
            mse_V = mean((V_pred .- y_sample[2]).^2)
            return mse_S + mse_V
        end
        val_loss += compute_loss(θ)
    end

    return val_loss / val_samples
end

# Train the model and collect loss history
train_loss_history, val_loss_history = train_model_batch_with_logging(
    x_train, y_train, x_test, y_test, θ_vec, epochs, batch_size, lr
)

# Plot training and validation losses
log("Plotting loss history...")
plot(1:epochs, train_loss_history, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Loss History")
plot!(1:epochs, val_loss_history, label="Validation Loss", xlabel="Epoch", ylabel="Loss", title="Loss History")
