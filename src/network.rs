use crate::matrix::Matrix;
use crate::traits::Layer;
use crate::loss::{MSE, CrossEntropy}; // CrossEntropy buraya eklendi

pub struct NeuralNetwork {
    pub pipeline: Vec<Box<dyn Layer>>,
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
    pub fn fit(
        &mut self,
        x_train: &Matrix,
        y_train: &Matrix,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> Vec<(f64, f64)> {
        let mut history = Vec::new();
        let total_samples = x_train.cols;

        for _epoch in 0..epochs {
            let mut start_idx = 0;
            while start_idx < total_samples {
                let end_idx = (start_idx + batch_size).min(total_samples);
                let x_batch = x_train.slice_cols(start_idx, end_idx);
                let y_batch = y_train.slice_cols(start_idx, end_idx);

                // 1. İleri Besleme
                let output = self.run(&x_batch);

                // 2. Geri Yayılım
                let mut gradient = CrossEntropy::derivative(&output, &y_batch);
                for layer in self.pipeline.iter_mut().rev() {
                    gradient = layer.backward(&gradient, learning_rate);
                }

                start_idx = end_idx;
            }

            // Epoch sonu değerlendirmesi
            let predictions = self.run(x_train);
            let loss = CrossEntropy::forward(&predictions, y_train);
            let acc = CrossEntropy::accuracy(&predictions, y_train);
            history.push((loss, acc));
        }

        history
    }
}
