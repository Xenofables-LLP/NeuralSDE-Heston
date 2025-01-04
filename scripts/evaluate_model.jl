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

log("Starting Neural SDE Model Testing...")

# === 1. Load model + training info from the saved file ===
model_file_path = "neural_sde_model.bson"
log("Loading model from: $model_file_path")
BSON.@load model_file_path θ_vec train_loss_history val_loss_history

log("Model loaded. θ_vec length = $(length(θ_vec))")
log("θ_vec stats => mean: $(mean(θ_vec)), std: $(std(θ_vec))")

# === 2. Rebuild neural networks with same layer dims as training ===
drift_layers = [(11, 32), (32, 32), (32, 2)]
diffusion_layers = [(11, 32), (32, 32), (32, 2)]

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
    l1, idx = build_dense(θ_val, idx, 11, 32, leakyrelu)
    l2, idx = build_dense(θ_val, idx, 32, 32, leakyrelu)
    l3, idx = build_dense(θ_val, idx, 32, 2, identity)
    return Chain(l1, l2, l3)
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

nn_drift, nn_diffusion = build_models(θ_vec)
log("Neural drift & diffusion models successfully rebuilt.")

# === 3. Load the dataset again & create the same test set ===
#    - We'll replicate the same process or just load from your training script's logic
file_path = "data/options-A-merged.csv"
dataset = CSV.File(file_path) |> DataFrame

train_ratio = 0.8
num_samples = size(dataset, 1)
num_train = Int(num_samples * train_ratio)

# Reuse the same feature extraction + normalization
# (Here we assume you didn't change the logic from the training script.)
function preprocess_data(dataset)
    # (Identical feature extraction from your training script)
    CLOSE = Float32.(dataset.CLOSE)
    vol   = Float32.(dataset.vol)
    delta = Float32.(dataset.delta)
    gamma = Float32.(dataset.gamma)
    theta = Float32.(dataset.theta)
    vega  = Float32.(dataset.vega)
    rho   = Float32.(dataset.rho)
    bid   = Float32.(dataset.bid)
    ask   = Float32.(dataset.ask)
    strike_normalized = Float32.(dataset.strike ./ dataset.CLOSE)
    call_put_binary   = Float32.((dataset.call_put .== "Call") .* 1 .+ (dataset.call_put .!= "Call") .* -1)

    midpoint = Float32.((bid + ask) ./ 2)
    T = Float32.(Dates.value.(dataset.expiration .- dataset.date)) ./ 365.0

    # Features
    features = hcat(
        CLOSE, vol, delta, gamma, theta,
        vega, rho, T, strike_normalized, midpoint,
        call_put_binary
    )

    # Calc global mean/std
    feature_mean = mean(features, dims=1)
    feature_std  = std(features, dims=1)

    normalized_features = (features .- feature_mean) ./ feature_std
    normalized_features = clamp.(normalized_features, -3.0f0, 3.0f0)

    return normalized_features, feature_mean, feature_std
end

all_features, _, _ = preprocess_data(dataset)

# Split into train/test the same way as the training script
x_test = all_features[num_train+1:end, :]
# Our target is the first 2 columns => (CLOSE, vol)
y_test = all_features[num_train+1:end, 1:2]

log("x_test size: $(size(x_test)), y_test size: $(size(y_test))")

# === 4. Forward pass to get predictions on test set ===
# We'll replicate the "forward_batch" approach, but for the entire test set in one pass
function forward_batch(x_batch, nn_drift, nn_diffusion; dt=1.0f0 / size(x_batch, 1))
    num_samples = size(x_batch, 1)

    dW1_values = sqrt(dt) .* randn(Float32, num_samples)
    dZ_values  = sqrt(dt) .* randn(Float32, num_samples)
    dW2_values = -0.7f0 .* dW1_values .+ sqrt(1 - (-0.7f0)^2) .* dZ_values

    S_pred = similar(dW1_values)
    V_pred = similar(dW2_values)

    for i in 1:num_samples
        f = nn_drift(x_batch[i, :])
        g = nn_diffusion(x_batch[i, :])
        fS, fV = f[1], f[2]
        gS, gV = g[1], g[2]

        V_next = max(0.0f0, fV * dt + gV * dW2_values[i])
        S_next = fS * dt + sqrt(max(0.0f0, V_next)) * gS * dW1_values[i]
        S_pred[i] = S_next
        V_pred[i] = V_next
    end
    return S_pred, V_pred
end

log("Running forward pass on x_test...")
S_pred, V_pred = forward_batch(x_test, nn_drift, nn_diffusion; dt=1.0f0 / size(x_test, 1))
log("Forward pass completed on test set.")

# === 5. Debug stats + plotting
log("S_pred => min=$(minimum(S_pred)), max=$(maximum(S_pred))")
log("V_pred => min=$(minimum(V_pred)), max=$(maximum(V_pred))")

# Reconstruct the same order of ground truth
S_true = y_test[:, 1]
V_true = y_test[:, 2]

mkpath("plots")  # ensure we have a folder for saving

# (a) Re-plot training vs validation loss (carried in train_loss_history/val_loss_history)
epochs = collect(1:length(train_loss_history))
plt_loss = plot(
    epochs, train_loss_history,
    label="Training Loss",
    xlabel="Epoch", ylabel="Loss",
    title="Loss History"
)
plot!(plt_loss, epochs, val_loss_history, label="Validation Loss")
savefig(plt_loss, "plots/loss_history.png")
log("Saved training/validation loss plot to plots/loss_history.png")

# (b) Scatter plot for S
plt_scatter_s = scatter(
    S_true, S_pred,
    title = "Predicted S vs. True S",
    xlabel = "True S (normalized)",
    ylabel = "Predicted S (normalized)",
    markersize=3,
    legend=false
)
savefig(plt_scatter_s, "plots/scatter_s.png")
log("Saved scatter_s.png under plots/")

# (c) Scatter plot for V
plt_scatter_v = scatter(
    V_true, V_pred,
    title = "Predicted V vs. True V",
    xlabel = "True V (normalized)",
    ylabel = "Predicted V (normalized)",
    markersize=3,
    legend=false
)
savefig(plt_scatter_v, "plots/scatter_v.png")
log("Saved scatter_v.png under plots/")

# (d) Optional: Histograms of residuals
resid_S = S_pred .- S_true
resid_V = V_pred .- V_true

plt_resid_S = histogram(
    resid_S, bins=50,
    title="Residuals in S (S_pred - S_true)",
    xlabel="Residual",
    ylabel="Count"
)
savefig(plt_resid_S, "plots/residuals_s.png")
log("Saved residuals_s.png under plots/")

plt_resid_V = histogram(
    resid_V, bins=50,
    title="Residuals in V (V_pred - V_true)",
    xlabel="Residual",
    ylabel="Count"
)
savefig(plt_resid_V, "plots/residuals_v.png")
log("Saved residuals_v.png under plots/")

log("All test plots saved. Inspect them to see if predictions align with true data.")
log("Testing script completed successfully.")
