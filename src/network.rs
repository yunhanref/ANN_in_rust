use crate::matrix::Matrix;
use crate::traits::Layer;

pub struct NeuralNetwork {
    // C++'taki std::vector<BaseLayer*> pipeline;
    pipeline: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork {
            pipeline: Vec::new(),
        }
    }

    // C++: void pushLayer(BaseLayer* l);
    pub fn push_layer(&mut self, layer: Box<dyn Layer>) {
        self.pipeline.push(layer);
    }

    // C++: Matrix run(const Matrix& input);
    pub fn run(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in &self.pipeline {
            current = layer.forward(&current);
        }
        current
    }
}