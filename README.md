# OOP-Brain: Rust GUI Edition

OOP-Brain is a robust, from-scratch feedforward neural network and matrix computation engine originally developed in C++ and systematically re-engineered into memory-safe Rust with an integrated graphical user interface.

<img width="80" height="60" alt="vid" src="https://github.com/user-attachments/assets/b5944cc0-b6ad-4b4c-b49d-c0800bd07865" />

## Architectural Overview


The system is engineered following modular software design principles, decoupling core numerical routines from user interface layers. The structural hierarchy is organized into the following core modules:

* **Matrix Engine (`matrix.rs`)**: Replaces raw pointer allocations with Rust's safe heap-allocated `Vec<f64>` vectors, supporting linear transformations, broadcasting operations, transpositions, and Hadamard products[cite: 17].
* **Activation Framework (`activations.rs`)**: Implements non-linear transformations and analytical derivatives for **Sigmoid**, **ReLU**, and **Tanh** functions[cite: 18].
* **Polymorphic Traits (`traits.rs`)**: Establishes core interfaces via `Activation` and `Layer` traits to enforce decoupled component interaction[cite: 19].
* **Neural Network Pipeline (`network.rs`)**: Manages sequential layer execution (`pipeline`), forward inference propagation, and gradient-based error optimization via `train_step`[cite: 20].
* **Error Management (`errors.rs`)**: Replaces traditional exceptions with a structured `BrainError` enum to ensure safe, panic-free error propagation[cite: 21].
* **Persistence Module (`io.rs`)**: Facilitates structured tabular dataset parsing and model weight persistence using comma-separated value (`.csv`) streams[cite: 22].
* **Graphical Interface (`main.rs`)**: Powered by the `eframe`/`egui` immediate-mode GUI framework to render real-time simulation metrics and interactive training controls[cite: 23].

## Mathematical Formulation & Backpropagation

The engine computes forward activations via affine transformations ($Z = W \cdot X + b$) followed by non-linear mapping. Gradient descent optimization updates synaptic weights using Mean Squared Error (MSE) loss derivatives propagated backward through the sequential pipeline.

## Getting Started

To compile and execute the desktop application locally, ensure Rust and Cargo are installed, then configure your workspace dependencies inside `Cargo.toml`:

```toml
[dependencies]
eframe = "0.24"
rand = "0.8"
winapi = { version = "0.3.9", features = ["winuser", "windef"] }
