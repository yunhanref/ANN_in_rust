# OOP-Brain: Rust GUI & ANN Edition

OOP-Brain is a robust, from-scratch feedforward neural network and matrix computation engine engineered in memory-safe Rust with an integrated graphical user interface (`eframe`/`egui`).

<img width="1086" height="599" alt="ann_rust" src="https://github.com/user-attachments/assets/23ce6f92-c25f-4d97-898b-d902d1a131bc" />

## Features

* **From-Scratch Architecture:** Implements matrix operations, linear algebra transformations, and backpropagation without relying on heavy deep-learning frameworks.
* **Multi-Class Classification:** Supports **ReLU**, **Sigmoid**, and **Softmax** activations paired with **Categorical Cross-Entropy** and **MSE** loss functions.
* **Tabular Data Processing:** Built-in CSV dataset parser with automated dataset splitting and shuffling mechanisms.
* **Interactive GUI Dashboard:** Real-time training parameter tuning (Epochs, Learning Rate, Batch Size), live loss plotting via `egui_plot`, and inference tools.
* **Model Persistence:** Save and load trained synaptic weights and biases securely.

## Project Structure

* `matrix.rs`:      Heap-allocated safe matrix math engine supporting transposition, broadcasting, and slicing.
* `activations.rs`: Mathematical activation functions and numerical stability routines.
* `traits.rs`:      Polymorphic core abstractions for `Layer` and `Activation`.
* `network.rs`:     Sequential pipeline execution and gradient descent optimization.
* `dense_layer.rs`: Fully connected linear layers with thread-safe interior mutability (`Mutex`).
* `loss.rs`:        Evaluation metrics, loss derivatives, and classification accuracy calculations.
* `dataset.rs`:     Data preprocessing, one-hot encoding, and train/test partitioning.
* `io.rs`:          Safe file stream management for CSV ingestion and model weight exports.
* `main.rs`:        Immediate-mode desktop graphical user interface.

## Getting Started

Ensure you have Rust and Cargo installed locally. Clone the repository and run the application via Cargo:

```bash
cargo run
```

```toml
[dependencies]
eframe = "0.24"
rand = "0.8"
winapi = { version = "0.3.9", features = ["winuser", "windef"] }
```
