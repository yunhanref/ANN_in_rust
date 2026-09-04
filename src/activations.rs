use crate::traits::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn transform(&self, val: f64) -> f64 {
        // C++: 1.0 / (1.0 + std::exp(-val))
        1.0 / (1.0 + (-val).exp())
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn transform(&self, val: f64) -> f64 {
        // C++: (val > 0.0) ? val : 0.0
        val.max(0.0) // Rust'ta f64 icin daha idiomatic bir yazim
    }
}

pub struct Tanh;

impl Activation for Tanh {
    fn transform(&self, val: f64) -> f64 {
        // C++: std::tanh(val)
        val.tanh()
    }
}