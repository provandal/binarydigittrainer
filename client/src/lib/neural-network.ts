// Simulated Neural Network for educational purposes
export class SimulatedNeuralNetwork {
  private weights: {
    inputToHidden: number[][];
    hiddenToOutput: number[][];
  };
  private biases: {
    hidden: number[];
    output: number[];
  };
  private learningRate: number = 0.001;

  constructor() {
    // Initialize with random weights
    this.weights = {
      inputToHidden: this.initializeWeights(64, 32),
      hiddenToOutput: this.initializeWeights(32, 10)
    };
    this.biases = {
      hidden: new Array(32).fill(0).map(() => Math.random() * 0.1),
      output: new Array(10).fill(0).map(() => Math.random() * 0.1)
    };
  }

  private initializeWeights(inputSize: number, outputSize: number): number[][] {
    return Array.from({ length: outputSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() - 0.5) * 0.2)
    );
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  private softmax(values: number[]): number[] {
    const max = Math.max(...values);
    const exp = values.map(v => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map(v => v / sum);
  }

  public forward(input: number[]): {
    hiddenActivations: number[];
    outputActivations: number[];
    predictions: number[];
  } {
    // Input to hidden layer
    const hiddenInputs = this.weights.inputToHidden.map((weights, i) =>
      weights.reduce((sum, weight, j) => sum + weight * input[j], this.biases.hidden[i])
    );
    const hiddenActivations = hiddenInputs.map(x => this.sigmoid(x));

    // Hidden to output layer
    const outputInputs = this.weights.hiddenToOutput.map((weights, i) =>
      weights.reduce((sum, weight, j) => sum + weight * hiddenActivations[j], this.biases.output[i])
    );
    const outputActivations = outputInputs.map(x => this.sigmoid(x));
    const predictions = this.softmax(outputActivations);

    return { hiddenActivations, outputActivations, predictions };
  }

  public train(input: number[], targetLabel: number): number {
    const target = new Array(10).fill(0);
    target[targetLabel] = 1;

    const { hiddenActivations, outputActivations, predictions } = this.forward(input);

    // Calculate loss (mean squared error)
    const loss = predictions.reduce((sum, pred, i) => 
      sum + Math.pow(pred - target[i], 2), 0) / 2;

    // Simulate backpropagation by slightly adjusting weights toward better performance
    this.simulateBackpropagation(input, hiddenActivations, predictions, target);

    return loss;
  }

  private simulateBackpropagation(
    input: number[], 
    hiddenActivations: number[], 
    predictions: number[], 
    target: number[]
  ): void {
    // Simulate weight updates with some randomness for educational effect
    const outputError = predictions.map((pred, i) => pred - target[i]);
    
    // Update output layer weights
    for (let i = 0; i < this.weights.hiddenToOutput.length; i++) {
      for (let j = 0; j < this.weights.hiddenToOutput[i].length; j++) {
        const gradient = outputError[i] * hiddenActivations[j];
        this.weights.hiddenToOutput[i][j] -= this.learningRate * gradient + 
          (Math.random() - 0.5) * 0.0001; // Add some noise for realism
      }
      this.biases.output[i] -= this.learningRate * outputError[i];
    }

    // Update hidden layer weights (simplified)
    for (let i = 0; i < this.weights.inputToHidden.length; i++) {
      for (let j = 0; j < this.weights.inputToHidden[i].length; j++) {
        const hiddenError = outputError.reduce((sum, err, k) => 
          sum + err * this.weights.hiddenToOutput[k][i], 0);
        const gradient = hiddenError * hiddenActivations[i] * (1 - hiddenActivations[i]) * input[j];
        this.weights.inputToHidden[i][j] -= this.learningRate * gradient * 0.1 + 
          (Math.random() - 0.5) * 0.0001;
      }
    }
  }

  public reset(): void {
    this.weights = {
      inputToHidden: this.initializeWeights(64, 32),
      hiddenToOutput: this.initializeWeights(32, 10)
    };
    this.biases = {
      hidden: new Array(32).fill(0).map(() => Math.random() * 0.1),
      output: new Array(10).fill(0).map(() => Math.random() * 0.1)
    };
  }

  public getLearningRate(): number {
    return this.learningRate;
  }
}
