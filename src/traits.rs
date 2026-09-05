use crate::matrix::Matrix;

pub trait Activation: Send + Sync {
    fn transform(&self, val: f64) -> f64;
    fn derivative(&self, val: f64) -> f64;
}
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_gradient: &Matrix, learning_rate: f64) -> Matrix;
}
