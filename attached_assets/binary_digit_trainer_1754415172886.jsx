### 🧠 Binary Digit Trainer: A Step-by-Step Neural Network Learning Simulator

**Binary Digit Trainer** is an interactive educational tool designed to teach the fundamentals of neural network training—step by step. Rather than simply recognizing digits, this tool walks users through the *entire learning process* of a small neural network, visualizing each stage of training in real time.

---

### 🎯 What It Does

This tool trains a simple feedforward neural network to distinguish between binary handwritten digits: **0** and **1**. Users draw digits on a 3×3 grid and label them as “0” or “1”, then step through the following process:

1. View the current weights and biases
2. Perform a **feedforward pass**
3. View activations of the hidden and output layers
4. Compute the **loss**
5. Perform **backpropagation**
6. Watch weights and biases **update visually**

Each training cycle is broken into these discrete steps so learners can see exactly what happens and why.

---

### 🧱 Neural Network Architecture

- **Input Layer**: 9 neurons (3×3 grid)
- **Hidden Layer**: 4 neurons
- **Output Layer**: 2 neurons (one for “0”, one for “1”)

All neurons are fully connected, using **sigmoid activation** and **mean squared error (MSE)** loss.

---

### 🖼 Visual Features

- **Interactive 3×3 input grid** to draw digits
- **Radio buttons** to select the correct label (0 or 1)
- **Per-neuron weight graphs** showing the strength and direction of each input connection
- **Live display of activations**, biases, and weight changes
- Visual feedback for loss and output predictions

---

### 🛠 Built With

- **React + TailwindCSS** for a clean and dynamic UI
- Designed for use in **Replit** with easy step-through interactions

---

### ✅ Ideal For

- Students learning how neural networks work
- Teachers or webinar hosts demoing machine learning concepts
- Engineers seeking a visual, intuitive feel for backpropagation

---

🔄 Want to reset, experiment, or modify the architecture? It’s open, hackable, and entirely yours to explore.
