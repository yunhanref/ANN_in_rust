use crate::matrix::Matrix;
use crate::traits::Layer;
use crate::loss::MSE;

pub struct NeuralNetwork {
    pipeline: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork {
            pipeline: Vec::new(),
        }
    }

    pub fn push_layer(&mut self, layer: Box<dyn Layer>) {
        self.pipeline.push(layer);
    }

    pub fn run(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in &self.pipeline {
            current = layer.forward(&current);
        }
        current
    }

    pub fn train_step(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64) -> f64 {
        let output = self.run(input);
        let loss = MSE::forward(&output, target);
        let mut gradient = MSE::derivative(&output, target);

        for layer in self.pipeline.iter_mut().rev() {
            gradient = layer.backward(&gradient, learning_rate);
        }

        loss
    }
}
