use crate::matrix::Matrix;

pub trait Activation: Send + Sync {
    fn transform(&self, val: f64) -> f64;
    fn derivative(&self, val: f64) -> f64;

    // Standart eleman bazli donusum (Sigmoid, ReLU, Tanh)
    fn apply(&self, z: &Matrix) -> Matrix {
        let mut a = Matrix::new(z.rows, z.cols);
        for i in 0..z.rows {
            for j in 0..z.cols {
                *a.get_mut(i, j) = self.transform(z.get(i, j));
            }
        }
        a
    }

    // Standart gradyan turev carpimi
    fn apply_derivative(&self, output_grad: &Matrix, z: &Matrix) -> Matrix {
        let mut sp = Matrix::new(z.rows, z.cols);
        for i in 0..z.rows {
            for j in 0..z.cols {
                *sp.get_mut(i, j) = self.derivative(z.get(i, j));
            }
        }
        output_grad.hadamard(&sp).unwrap()
    }
}
pub trait Layer: Send + Sync {
    fn forward(&self, input: &Matrix) -> Matrix;
    fn backward(&mut self, output_gradient: &Matrix, learning_rate: f64) -> Matrix;
    fn get_params(&self) -> Option<(Matrix, Matrix)> { None }
    fn set_params(&mut self, _w: &Matrix, _b: &Matrix) -> Result<(), String> { Err("Desteklenmiyor".to_string()) }	
}