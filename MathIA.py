# cities_softmax_planes.py
# Single-layer softmax classifier for four cities
# and final parameters in plane form: z = w_x * lat + w_y * lon + b

import numpy as np

# -------------------------
# Raw data: lat, lon for 4 cities
# -------------------------
city_names = ["Philadelphia", "Cleveland", "Pittsburgh", "New York City"]

coords_deg = np.array([
    [-75.162989,   39.9524],   # Barcelona
    [-81.69355,   41.505089],   # Paris
    [-79.997543,   40.438385],   # Madrid
    [-74.003387,   40.714119],   # Berlin
], dtype=np.float32)

# One-hot labels
y_true = np.eye(4, dtype=np.float32)   # shape (4, 4)

# -------------------------
# Normalize inputs for training
# -------------------------
X_raw = coords_deg.copy()  # keep original for later
X_mean = X_raw.mean(axis=0, keepdims=True)  # (1, 2)
X_std  = X_raw.std(axis=0, keepdims=True)   # (1, 2)

X = (X_raw - X_mean) / (X_std + 1e-8)       # (4, 2)


# -------------------------
# Model: X_norm -> logits -> softmax
# -------------------------
def softmax(logits):
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X_norm, W, b):
    logits = X_norm @ W + b         # (N, 4)
    probs = softmax(logits)         # (N, 4)
    return probs, logits

def cross_entropy(probs, y_true):
    eps = 1e-9
    correct = np.sum(y_true * probs, axis=1)  # (N,)
    return -np.mean(np.log(correct + eps))

def backward(X_norm, probs, y_true):
    N = X_norm.shape[0]
    dlogits = (probs - y_true) / N          # (N, 4)

    dW = X_norm.T @ dlogits                 # (2, 4)
    db = np.sum(dlogits, axis=0, keepdims=True)   # (1, 4)
    return dW, db


# -------------------------
# Training
# -------------------------
def train_softmax(
    X_norm,
    y_true,
    num_steps=10000000000,
    learning_rate=0.00001,
    print_every=500,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # Initialize parameters in normalized space
    W = rng.normal(0, 0.1, size=(2, 4)).astype(np.float32)
    b = np.zeros((1, 4), dtype=np.float32)

    W_init = W.copy()
    b_init = b.copy()

    for step in range(1, num_steps + 1):
        probs, logits = forward(X_norm, W, b)
        loss = cross_entropy(probs, y_true)
        dW, db = backward(X_norm, probs, y_true)

        W -= learning_rate * dW
        b -= learning_rate * db

        if step == 1 or step % print_every == 0 or step == num_steps:
            preds = np.argmax(probs, axis=1)
            true = np.argmax(y_true, axis=1)
            acc = np.mean(preds == true)
            print(f"Step {step:5d} | loss={loss:.4f} | acc={acc:.3f}")

    return W, b, W_init, b_init


# -------------------------
# Convert to raw-plane form: z = w_x * lat + w_y * lon + b
# -------------------------
def params_to_raw_planes(W_norm, b_norm, X_mean, X_std):
    """
    W_norm, b_norm are in normalized coordinates:
      z = W_norm^T * X_norm + b_norm

    X_norm = (X_raw - mean) / std

    We want planes in raw coordinates:
      z = w_x * lat + w_y * lon + b

    Derivation:
      X_norm = (X_raw - mean) / std
      z = W_norm^T * ((X_raw - mean) / std) + b_norm
        = (W_norm / std)^T * X_raw + (b_norm - W_norm^T * mean / std)

    So:
      W_raw = W_norm / std
      b_raw = b_norm - (mean / std) @ W_norm
    """
    # X_mean, X_std: (1,2)
    # W_norm: (2,4), b_norm: (1,4)
    std = X_std  # (1,2)
    mean = X_mean

    # Divide each row of W by std components
    # W_raw[i, k] = W_norm[i, k] / std[i]
    W_raw = W_norm / (std.T + 1e-8)   # (2,4)

    # Compute mean/std dot W_norm: (1,2) / (1,2) -> (1,2) then @ (2,4) -> (1,4)
    mean_over_std = mean / (std + 1e-8)      # (1,2)
    correction = mean_over_std @ W_norm      # (1,4)

    b_raw = b_norm - correction              # (1,4)

    return W_raw, b_raw


# -------------------------
# Utility: prediction and pretty printing
# -------------------------
def predict_probs_for_cities(W_norm, b_norm):
    probs, logits = forward(X, W_norm, b_norm)
    return probs  # (4,4) rows = cities, cols = city class

def print_planes(W_raw, b_raw):
    # W_raw shape (2,4), b_raw (1,4)
    print("\n=== Final planes in raw lat/lon ===")
    for k, name in enumerate(city_names):
        wx = W_raw[0, k]
        wy = W_raw[1, k]
        bk = b_raw[0, k]
        print(f"{name}: z = {wx:.6f} * lat + {wy:.6f} * lon + {bk:.6f}")

def print_probs(probs):
    print("\n=== Final probabilities at each city's coordinates ===")
    # probs: (4,4)
    for i, name in enumerate(city_names):
        row = probs[i]
        print(f"{name:10s} -> ", end="")
        for k, cname in enumerate(city_names):
            print(f"{cname[0:3]}: {row[k]:.4f}  ", end="")
        print("")


# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    # You can tune these:
    num_steps = 5000
    learning_rate = 0.05

    print(f"Training with {num_steps} steps, lr={learning_rate}...\n")
    W_final, b_final, W_init, b_init = train_softmax(
        X,
        y_true,
        num_steps=num_steps,
        learning_rate=learning_rate,
        print_every=max(1, num_steps // 10),
        seed=0,
    )

    # 1. Initial parameters in normalized space
    print("\n=== INITIAL PARAMETERS (normalized space) ===")
    print("W_init (shape {}):".format(W_init.shape))
    print(W_init)
    print("\nb_init (shape {}):".format(b_init.shape))
    print(b_init)

    # 2. Final parameters in normalized space
    print("\n=== FINAL PARAMETERS (normalized space) ===")
    print("W_final (shape {}):".format(W_final.shape))
    print(W_final)
    print("\nb_final (shape {}):".format(b_final.shape))
    print(b_final)

    # 3. Convert to raw-coordinate planes z = w_x * lat + w_y * lon + b
    W_raw, b_raw = params_to_raw_planes(W_final, b_final, X_mean, X_std)
    print_planes(W_raw, b_raw)

    # 4. Final probabilities for each city at its own coordinates
    probs_final = predict_probs_for_cities(W_final, b_final)
    print_probs(probs_final)

