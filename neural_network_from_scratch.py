"""

1. WEIGHTS AND BIASES
   - Weights (W): Represent strength of connections between layers. They scale the
     signals. In a linear step: Z = XW + b.
   - Biases (b): Allow shifting the activation function left/right to ensure neurons
     can fire (or not fire) independently of input scale.

2. FORWARD PROPAGATION
   For layer l with input A^(l-1), weights W^(l), and bias b^(l):
     Z^(l) = A^(l-1) * W^(l) + b^(l)  (Linear combination)
     A^(l) = g(Z^(l))                 (Activation function)

   Our Architecture: 4 Inputs -> Hidden 1 (8 ReLU) -> Hidden 2 (4 ReLU) -> Output (1 Sigmoid)
     - Input layer:   A^(0) = X (size: m x 4)
     - Hidden 1:      Z^(1) = X * W^(1) + b^(1)          (W1 size: 4 x 8, b1 size: 1 x 8)
                      A^(1) = ReLU(Z^(1))
     - Hidden 2:      Z^(2) = A^(1) * W^(2) + b^(2)      (W2 size: 8 x 4, b2 size: 1 x 4)
                      A^(2) = ReLU(Z^(2))
     - Output layer:  Z^(3) = A^(2) * W^(3) + b^(3)      (W3 size: 4 x 1, b3 size: 1 x 1)
                      A^(3) = Sigmoid(Z^(3)) = y_pred

3. ACTIVATION FUNCTIONS
   - ReLU (Rectified Linear Unit):
     g(z) = max(0, z)
     Derivative: g'(z) = 1 if z > 0 else 0
     *Why hidden layers use ReLU:* Mitigates the vanishing gradient problem, computes 
     extremely fast, and introduces sparsity.
   - Sigmoid:
     g(z) = 1 / (1 + e^-z)
     Derivative: g'(z) = g(z) * (1 - g(z))
     *Why output layer uses Sigmoid:* Squashes any real value into (0, 1), representing
     a probability for binary classification.

4. LOSS FUNCTION: BINARY CROSS-ENTROPE
   Quantifies the difference between target probability y and predicted probability y_pred.
     Loss = - (1/m) * sum( y * ln(y_pred) + (1 - y) * ln(1 - y_pred) )

5. BACKPROPAGATION (Gradient Flow via Chain Rule)
   Goal: Find partial derivatives of Loss (L) with respect to W and b in each layer.
   For Output Layer (layer 3 with Sigmoid & BCE loss):
     dZ^(3) = A^(3) - y      <-- Key simplification for Sigmoid + Cross-Entropy loss!
     dW^(3) = (1/m) * (A^(2))^T * dZ^(3)
     db^(3) = (1/m) * sum(dZ^(3), axis=0)

   For Hidden Layer 2 (layer 2 with ReLU):
     dA^(2) = dZ^(3) * (W^(3))^T
     dZ^(2) = dA^(2) * g'(Z^(2))  (where * is element-wise multiplication)
     dW^(2) = (1/m) * (A^(1))^T * dZ^(2)
     db^(2) = (1/m) * sum(dZ^(2), axis=0)

   For Hidden Layer 1 (layer 1 with ReLU):
     dA^(1) = dZ^(2) * (W^(2))^T
     dZ^(1) = dA^(1) * g'(Z^(1))
     dW^(1) = (1/m) * X^T * dZ^(1)
     db^(1) = (1/m) * sum(dZ^(1), axis=0)

6. GRADIENT DESCENT (Optimization Step)
   Updates parameters in the opposite direction of the gradient to minimize the loss:
     W^(l) = W^(l) - alpha * dW^(l)
     b^(l) = b^(l) - alpha * db^(l)   (where alpha is the learning rate)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Matrix Multiplication
# ------------------------------------------------------------------------------
def matrix_multiply(A, B, method='vectorized'):
    """
    Multiplies 2D matrices A (M x K) and B (K x N) using custom NumPy code.
    
    Methods:
      - 'vectorized': Uses NumPy broadcasting to perform multiplication and summation
                      without np.dot/@. Much faster (under 0.1s for training).
      - 'loop': Uses a manual triple-nested Python loop. Slow (1-2 minutes for training),
                but serves as a direct translation of the mathematical formula.
      - 'numpy': Uses native NumPy matmul (A @ B).
    """
    A = np.asarray(A)
    B = np.asarray(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both inputs must be 2D matrices.")
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Dimension mismatch: {A.shape} cannot multiply {B.shape}")

    if method == 'vectorized':
        # Vectorized equivalent of the nested loop:
        # A[:, :, np.newaxis] is shape (M, K, 1)
        # B[np.newaxis, :, :] is shape (1, K, N)
        # Broadcasting gives shape (M, K, N) where product[i, k, j] = A[i, k] * B[k, j]
        # Summing over the K dimension (axis=1) yields the final shape (M, N)
        return np.sum(A[:, :, np.newaxis] * B[np.newaxis, :, :], axis=1)
        
    elif method == 'numpy':
        return A @ B

    elif method == 'loop':
        result = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    result[i, j] += A[i, k] * B[k, j]
        return result

    else:
        raise ValueError(f"Unknown multiplication method: {method}")


# ------------------------------------------------------------------------------
# 2. Activation Functions and Derivatives
# ------------------------------------------------------------------------------
def relu(z):
    """Computes Rectified Linear Unit: max(0, z)"""
    return np.maximum(0, z)


def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0 else 0"""
    return (z > 0).astype(float)


def sigmoid(z):
    """Computes Sigmoid activation function with clipping to prevent overflow."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------------------
# 3. Loss Function
# ------------------------------------------------------------------------------
def binary_cross_entropy(y_true, y_pred):
    """Computes Binary Cross Entropy Loss between true labels and predictions."""
    eps = 1e-15  # Tiny epsilon to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


# ------------------------------------------------------------------------------
# 4. Neural Network Class
# ------------------------------------------------------------------------------
class NeuralNetwork:
    """
    A 3-layer neural network (4 inputs -> 8 hidden -> 4 hidden -> 1 output)
    implemented from scratch using NumPy.
    """
    def __init__(self, input_size=4, hidden1=8, hidden2=4, output_size=1, 
                 matrix_method='vectorized', seed=42):
        self.matrix_method = matrix_method
        rng = np.random.default_rng(seed)

        # He Initialization for layers using ReLU activation
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden1))
        self.b1 = np.zeros((1, hidden1))

        self.W2 = rng.normal(0, np.sqrt(2 / hidden1), (hidden1, hidden2))
        self.b2 = np.zeros((1, hidden2))

        # Xavier Initialization for output layer using Sigmoid activation
        self.W3 = rng.normal(0, np.sqrt(1 / hidden2), (hidden2, output_size))
        self.b3 = np.zeros((1, output_size))

        self.loss_history = []

    def forward(self, X):
        """Performs forward propagation through the network."""
        # Layer 1 (Hidden 1): Linear -> ReLU
        self.Z1 = matrix_multiply(X, self.W1, method=self.matrix_method) + self.b1
        self.A1 = relu(self.Z1)

        # Layer 2 (Hidden 2): Linear -> ReLU
        self.Z2 = matrix_multiply(self.A1, self.W2, method=self.matrix_method) + self.b2
        self.A2 = relu(self.Z2)

        # Layer 3 (Output): Linear -> Sigmoid
        self.Z3 = matrix_multiply(self.A2, self.W3, method=self.matrix_method) + self.b3
        self.A3 = sigmoid(self.Z3)

        return self.A3

    def backward(self, X, y, learning_rate):
        """Performs backpropagation and gradient descent parameter updates."""
        m = X.shape[0]

        # 1. Output Layer Gradients
        dZ3 = self.A3 - y
        dW3 = matrix_multiply(self.A2.T, dZ3, method=self.matrix_method) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # 2. Hidden Layer 2 Gradients
        dA2 = matrix_multiply(dZ3, self.W3.T, method=self.matrix_method)
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = matrix_multiply(self.A1.T, dZ2, method=self.matrix_method) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 3. Hidden Layer 1 Gradients
        dA1 = matrix_multiply(dZ2, self.W2.T, method=self.matrix_method)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = matrix_multiply(X.T, dZ1, method=self.matrix_method) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 4. Gradient Descent Weight & Bias Updates
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=2000, learning_rate=0.05, print_every=200):
        """Trains the model using backpropagation and gradient descent."""
        print(f"Training started using matrix multiplication method: '{self.matrix_method}'")
        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            self.backward(X, y, learning_rate)

            if epoch == 1 or epoch % print_every == 0:
                predictions = (y_pred >= 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f} | Training Accuracy: {accuracy * 100:.2f}%")

    def predict_probability(self, X):
        """Predicts the probability output (value between 0 and 1)."""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Predicts binary classification labels (0 or 1) based on a threshold."""
        prob = self.predict_probability(X)
        return (prob >= threshold).astype(int)


# ------------------------------------------------------------------------------
# 5. Helper Evaluation Function
# ------------------------------------------------------------------------------
def evaluate_metrics(y_true, y_pred):
    """
    Computes classification performance metrics: Accuracy, Precision, Recall,
    F1-Score, and confusion matrix components.
    """
    accuracy = np.mean(y_pred == y_true)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def train_test_split(X, y, test_size=0.20, seed=42):
    """Splits features and labels into training and test datasets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    test_count = int(len(X) * test_size)
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train, X_test):
    """
    Standardizes training and test features. Computes mean and standard deviation
    on training data and uses them to scale both splits.
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    
    return X_train_scaled, X_test_scaled, mean, std


# ------------------------------------------------------------------------------
# 6. Main Execution Pipeline
# ------------------------------------------------------------------------------
def main():
    # Resolve paths relative to this script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "students.csv")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    features = ["study_hours", "attendance", "previous_marks", "assignment_scores"]
    X = df[features].to_numpy(dtype=float)
    y = df["passed"].to_numpy(dtype=float).reshape(-1, 1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, seed=42)
    
    # Standardize data
    X_train_scaled, X_test_scaled, mean, std = standardize_train_test(X_train, X_test)
    
    # Instantiate neural network using the fast vectorized multiplication method
    # Change matrix_method to 'loop' to test the slow nested Python loops version
    model = NeuralNetwork(input_size=4, hidden1=8, hidden2=4, output_size=1, 
                          matrix_method='vectorized', seed=42)
    
    # Train model
    model.train(X_train_scaled, y_train, epochs=2500, learning_rate=0.08, print_every=250)
    
    # Make test predictions
    test_probs = model.predict_probability(X_test_scaled)
    test_preds = (test_probs >= 0.5).astype(int)
    
    # Calculate performance metrics
    metrics = evaluate_metrics(y_test, test_preds)
    
    print("\n" + "="*45)
    print("              MODEL PERFORMANCE EVALUATION")
    print("="*45)
    print(f"Test Set Samples : {len(y_test)}")
    print(f"Accuracy         : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision        : {metrics['precision'] * 100:.2f}%")
    print(f"Recall           : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score         : {metrics['f1_score'] * 100:.2f}%")
    print("-"*45)
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                 Fail     Pass")
    print(f"Actual Fail     {metrics['tn']:4d}     {metrics['fp']:4d}")
    print(f"Actual Pass     {metrics['fn']:4d}     {metrics['tp']:4d}")
    print("="*45)
    
    # Test Prediction on a New Student
    # Input format: [Study Hours, Attendance, Previous Marks, Assignment Scores]
    new_student = np.array([[7.0, 92.0, 78.0, 85.0]])
    new_student_scaled = (new_student - mean) / std
    
    probability = model.predict_probability(new_student_scaled)[0, 0]
    prediction = "Pass" if probability >= 0.5 else "Fail"
    
    print("\n" + "="*45)
    print("             DEMONSTRATION: NEW STUDENT PREDICTION")
    print("="*45)
    print("Inputs:")
    print("  - Study Hours        : 7.0 hours")
    print("  - Attendance         : 92.0%")
    print("  - Previous Marks     : 78.0%")
    print("  - Assignment Scores  : 85.0%")
    print("-"*45)
    print(f"Prediction Probability : {probability * 100:.2f}%")
    print(f"Class Prediction       : Student will {prediction.upper()}")
    print("="*45)
    
    # Plot Training Loss using premium aesthetics
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot the curve with a sleek dark-blue/indigo theme
    plt.plot(model.loss_history, color='#4f46e5', linewidth=2.5, label='Binary Cross-Entropy Loss')
    
    plt.title("Neural Network Loss Convergence Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epochs", fontsize=11, labelpad=10)
    plt.ylabel("Training Loss", fontsize=11, labelpad=10)
    
    # Style customization
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#94a3b8')
    ax.spines['bottom'].set_color('#94a3b8')
    ax.set_facecolor('#f8fafc')
    plt.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    
    # Info panel box
    stats_text = (
        f"Epochs: {len(model.loss_history)}\n"
        f"Initial Loss: {model.loss_history[0]:.4f}\n"
        f"Final Loss: {model.loss_history[-1]:.4f}\n"
        f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )
    props = dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor='#e2e8f0', alpha=0.9)
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.76), frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plot_path = os.path.join(script_dir, "training_loss.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nPremium training loss plot saved to '{plot_path}'.")
    plt.close()


if __name__ == "__main__":
    main()
