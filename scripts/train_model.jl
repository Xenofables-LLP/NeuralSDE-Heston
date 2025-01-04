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

function log(message::String)
    Zygote.ignore() do
        timestamp = Dates.format(now(), "yyyy-mm-dd HH:MM:SS.sss")
        println("[$timestamp] $message")
    end
end

log("Starting Functional Neural SDE with Zygote...")

# Load and Preprocess Dataset
file_path = "data/options-A-merged.csv"
log("Loading dataset from '$file_path'...")
dataset = CSV.File(file_path, header=true) |> DataFrame
log("Dataset loaded successfully!")

function preprocess_data(dataset, train_ratio=0.8)
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

    feature_mean = mean(features, dims=1)
    feature_std = std(features, dims=1)
    normalized_features = (features .- feature_mean) ./ feature_std
    normalized_features = clamp.(normalized_features, -3.0f0, 3.0f0)

    num_train = Int(size(dataset, 1) * train_ratio)
    x_train = normalized_features[1:num_train, :]
    y_train = normalized_features[1:num_train, 1:2]
    x_test = normalized_features[num_train+1:end, :]
    y_test = normalized_features[num_train+1:end, 1:2]

    return Float32.(x_train), Float32.(y_train), Float32.(x_test), Float32.(y_test), feature_mean, feature_std
end

x_train, y_train, x_test, y_test, feature_mean, feature_std = preprocess_data(dataset)
log("Preprocessed dataset.")

# Xavier initialization
function xavier_initialization(layers)
    θ_vals = []
    for (in_dim, out_dim) in layers
        W = Float32.(randn(out_dim, in_dim) .* sqrt(2.0 / (in_dim + out_dim)))
        b = Float32.(zeros(out_dim))
        push!(θ_vals, vec(W), b)
    end
    return reduce(vcat, θ_vals)
end

drift_layers = [(10, 32), (32, 32), (32, 2)]
diffusion_layers = [(10, 32), (32, 32), (32, 2)]

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
    drift_param_count = sum([in_dim * out_dim + out_dim for (in_dim, out_dim) in drift_layers])
    diff_param_count = sum([in_dim * out_dim + out_dim for (in_dim, out_dim) in diffusion_layers])

    drift_params = θ[1:drift_param_count]
    diff_params = θ[drift_param_count+1:end]

    nn_drift_built = build_chain(drift_params)
    nn_diffusion_built = build_chain(diff_params)
    return nn_drift_built, nn_diffusion_built
end

function forward_batch(x, θ; dt=1.0f0 / size(x, 1))
    nn_drift_built, nn_diffusion_built = build_models(θ)
    num_samples = size(x, 1)

    dW1_values = sqrt(dt) .* randn(Float32, num_samples)
    dZ_values = sqrt(dt) .* randn(Float32, num_samples)
    dW2_values = -0.7f0 .* dW1_values .+ sqrt(1 - (-0.7f0)^2) .* dZ_values

    S_values = map(i -> begin
        f = nn_drift_built(x[i, :])
        g = nn_diffusion_built(x[i, :])
        fS, fV = f[1], f[2]
        gS, gV = g[1], g[2]
        V_next = max(0.0f0, fV * dt + gV * dW2_values[i])
        S_next = fS * dt + sqrt(max(0.0f0, V_next)) * gS * dW1_values[i]
        (S_next, V_next)
    end, 1:num_samples)

    return map(first, S_values), map(last, S_values)
end

function calculate_loss(x, y, θ)
    S_pred, V_pred = forward_batch(x, θ)
    mse_S = mean((S_pred .- y[:, 1]).^2)
    mse_V = mean((V_pred .- y[:, 2]).^2)
    return mse_S + mse_V
end

function train_model(x_train, y_train, x_test, y_test, θ, epochs, lr)
    log("Starting training...")
    train_loss_history = []
    val_loss_history = []

    for epoch in 1:epochs
        function compute_loss(θ)
            return calculate_loss(x_train, y_train, θ) + 1e-4 * sum(θ.^2)
        end

        train_loss, pullback_fn = Zygote.pullback(compute_loss, θ)
        grad = pullback_fn(1)[1]
        clipped_grad = clamp.(grad, -10.0f0, 10.0f0)
        θ .= θ .- lr .* clipped_grad

        push!(train_loss_history, train_loss)
        val_loss = calculate_loss(x_test, y_test, θ)
        push!(val_loss_history, val_loss)

        log("Epoch $epoch: Training Loss = $train_loss, Validation Loss = $val_loss")
    end

    BSON.@save "neural_sde_model_nobatches.bson" θ=θ train_loss_history=train_loss_history val_loss_history=val_loss_history
    log("Model saved.")
    return train_loss_history, val_loss_history
end

θ_vec = xavier_initialization(drift_layers) + xavier_initialization(diffusion_layers)
epochs = 10
lr = 0.01f0
train_loss_history, val_loss_history = train_model(x_train, y_train, x_test, y_test, θ_vec, epochs, lr)

# Plot loss history
log("Plotting loss history...")
plot(1:epochs, train_loss_history, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Loss History")
plot!(1:epochs, val_loss_history, label="Validation Loss", xlabel="Epoch", ylabel="Loss", title="Loss History")
