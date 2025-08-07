import json
import math
import random
import argparse

# Activation functions
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Initialize weights and biases
def init_weights(n_in, n_out):
    return [[random.uniform(-0.5, 0.5) for _ in range(n_in)] for _ in range(n_out)]

def init_biases(n_out):
    return [random.uniform(-0.5, 0.5) for _ in range(n_out)]

# Feedforward pass
def feedforward(x, weights1, biases1, weights2, biases2, debug=False):
    z1 = [sum(w_ij * x_j for w_ij, x_j in zip(w_i, x)) + b_i for w_i, b_i in zip(weights1, biases1)]
    a1 = [sigmoid(z) for z in z1]
    z2 = [sum(w_ij * a1_j for w_ij, a1_j in zip(w_i, a1)) + b_i for w_i, b_i in zip(weights2, biases2)]
    a2 = [sigmoid(z) for z in z2]
    if debug:
        print("  Hidden activations:", a1)
        print("  Output activations:", a2)
    return z1, a1, z2, a2

# Backpropagation
def backpropagate(x, y, weights1, biases1, weights2, biases2, lr, debug=False):
    z1, a1, z2, a2 = feedforward(x, weights1, biases1, weights2, biases2, debug)
    delta2 = [(a2_i - y_i) * sigmoid_derivative(z2_i) for a2_i, y_i, z2_i in zip(a2, y, z2)]
    delta1 = []
    for i in range(len(weights1)):
        error = sum(delta2[j] * weights2[j][i] for j in range(len(weights2)))
        delta1.append(error * sigmoid_derivative(z1[i]))
    if debug:
        print("  Output deltas:", delta2)
        print("  Hidden deltas:", delta1)
    for i in range(len(weights2)):
        for j in range(len(weights2[i])):
            weights2[i][j] -= lr * delta2[i] * a1[j]
        biases2[i] -= lr * delta2[i]
    for i in range(len(weights1)):
        for j in range(len(weights1[i])):
            weights1[i][j] -= lr * delta1[i] * x[j]
        biases1[i] -= lr * delta1[i]

# Predict
def predict(x, weights1, biases1, weights2, biases2):
    _, _, _, a2 = feedforward(x, weights1, biases1, weights2, biases2)
    return a2.index(max(a2))

# Main training and testing loop
def train_and_test(debug=False):
    with open("binary_digit_training_set.json", "r") as f:
        dataset = json.load(f)

    input_size = 81
    hidden_size = 24
    output_size = 2
    learning_rate = 0.5
    epochs = 10

    weights1 = init_weights(input_size, hidden_size)
    biases1 = init_biases(hidden_size)
    weights2 = init_weights(hidden_size, output_size)
    biases2 = init_biases(output_size)

    for epoch in range(epochs):
        random.shuffle(dataset)
        print(f"Epoch {epoch+1}/{epochs}")
        for sample_index, sample in enumerate(dataset):
            x = sample["input"]
            y = sample["target"]
            if debug:
                print(f"Sample {sample_index+1}: Target = {y}")
            backpropagate(x, y, weights1, biases1, weights2, biases2, learning_rate, debug)

    correct = 0
    test_samples = random.sample(dataset, 10)
    for sample in test_samples:
        x = sample["input"]
        y = sample["target"]
        pred = predict(x, weights1, biases1, weights2, biases2)
        actual = y.index(1)
        print(f"Predicted: {pred}, Actual: {actual}")
        if pred == actual:
            correct += 1

    print(f"Accuracy: {correct / len(test_samples):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    train_and_test(debug=args.debug)
