# cities_softmax_backprop.py
# Single-layer softmax model for 4 cities:
# Barcelona, Paris, Madrid, Berlin
# No hidden layer, no ReLU. Just linear -> softmax.

import numpy as np

# -------------------------
# Data: lat, lon for 4 cities
# -------------------------
city_names = ["Barcelona", "Paris", "Madrid", "Berlin"]

coords_deg = np.array([
    [41.3851,   2.1734],   # Barcelona
    [48.8566,   2.3522],   # Paris
    [40.4168,  -3.7038],   # Madrid
    [52.5200,  13.4050],   # Berlin
], dtype=np.float32)

# One-hot labels for the 4 cities
y_true = np.eye(4, dtype=np.float32)   # shape (4, 4)

# Normalize inputs for better optimization
X = coords_deg.copy()
X_mean = X.mean(axis=0, keepdims=True)
X_std  = X.std(axis=0, keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)      # shape (4, 2)


# -------------------------
# Model: X (N, 2) → logits (N, 4) → softmax probs (N, 4)
# -------------------------
def softmax(logits):
    # logits: (N, 4)
    shift = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(X, W, b):
    # W: (2, 4), b: (1, 4)
    logits = X @ W + b
    probs = softmax(logits)
    return probs, logits

def cross_entropy(probs, y_true):
    # probs, y_true: (N, 4)
    eps = 1e-9
    correct_probs = np.sum(y_true * probs, axis=1)  # (N,)
    loss = -np.mean(np.log(correct_probs + eps))
    return loss

def backward(X, probs, y_true):
    """
    Derivatives for softmax + cross-entropy with respect to W and b
    for a single linear layer.

    X:     (N, 2)
    probs: (N, 4)
    y_true:(N, 4)
    returns dW (2, 4), db (1, 4)
    """
    N = X.shape[0]
    # dL/dlogits = (probs - y_true)/N for each example
    dlogits = (probs - y_true) / N  # (N, 4)

    # logits = X @ W + b
    dW = X.T @ dlogits              # (2, 4)
    db = np.sum(dlogits, axis=0, keepdims=True)  # (1, 4)

    return dW, db


# -------------------------
# Training
# -------------------------
def train_softmax(
    X,
    y_true,
    num_steps=100000000,
    learning_rate=0.00001,
    print_every=100,
    seed=42,
):
    """
    Train a single-layer softmax classifier via gradient descent.

    Returns:
      W_final, b_final, W_init, b_init
    """
    rng = np.random.default_rng(seed)

    # Initialize parameters (small random weights, zero bias)
    W = rng.normal(0, 0.1, size=(2, 4)).astype(np.float32)
    b = np.zeros((1, 4), dtype=np.float32)

    # Keep copies of initial parameters
    W_init = W.copy()
    b_init = b.copy()

    for step in range(1, num_steps + 1):
        probs, logits = forward(X, W, b)
        loss = cross_entropy(probs, y_true)

        dW, db = backward(X, probs, y_true)

        # Gradient descent update
        W -= learning_rate * dW
        b -= learning_rate * db

        if step == 1 or step % print_every == 0 or step == num_steps:
            preds = np.argmax(probs, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            acc = np.mean(preds == true_labels)
            print(f"Step {step:4d} | loss={loss:.4f} | acc={acc:.3f}")

    return W, b, W_init, b_init


def predict_city(lat, lon, W, b):
    x = np.array([[lat, lon]], dtype=np.float32)
    x_norm = (x - X_mean) / (X_std + 1e-8)
    probs, _ = forward(x_norm, W, b)
    idx = int(np.argmax(probs, axis=1)[0])
    return city_names[idx], probs[0]


if __name__ == "__main__":
    # Customize these two values:
    num_steps     = 100000000    # e.g. 500, 1000, 5000, ...
    learning_rate = 0.00001     # e.g. 0.01, 0.05, 0.1, ...

    print("Training with "
          f"{num_steps} steps, learning_rate={learning_rate}...\n")

    W_final, b_final, W_init, b_init = train_softmax(
        X,
        y_true,
        num_steps=num_steps,
        learning_rate=learning_rate,
        print_every=max(1, num_steps // 10),
    )

    # Print initial parameters
    print("\n=== INITIAL PARAMETERS ===")
    print("W_init (shape {}):".format(W_init.shape))
    print(W_init)
    print("\nb_init (shape {}):".format(b_init.shape))
    print(b_init)

    # Print final parameters
    print("\n=== FINAL PARAMETERS ===")
    print("W_final (shape {}):".format(W_final.shape))
    print(W_final)
    print("\nb_final (shape {}):".format(b_final.shape))
    print(b_final)

    # Check predictions on training points
    print("\n=== Predictions on training coordinates ===")
    for i, (lat, lon) in enumerate(coords_deg):
        city_pred, probs = predict_city(lat, lon, W_final, b_final)
        print(f"Input: ({lat:.4f}, {lon:.4f}) "
              f"True: {city_names[i]:10s} "
              f"Pred: {city_pred:10s} "
              f"Probs: {probs.round(3)}")
