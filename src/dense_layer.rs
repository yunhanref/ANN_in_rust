use crate::matrix::Matrix;
use crate::traits::{Layer, Activation};
use std::sync::Mutex;

pub struct DenseLayer {
    pub weights: Matrix,
    pub bias: Matrix,
    act_func: Box<dyn Activation>,
    // RefCell yerine thread-safe olan Mutex kullaniyoruz
    last_input: Mutex<Option<Matrix>>,
    last_z: Mutex<Option<Matrix>>,
}

impl DenseLayer {
    pub fn new(in_dim: usize, out_dim: usize, act_func: Box<dyn Activation>) -> Self {
        let mut weights = Matrix::new(out_dim, in_dim);
        let mut bias = Matrix::new(out_dim, 1);

        weights.randomize(-1.0, 1.0);
        bias.randomize(-1.0, 1.0);

        DenseLayer {
            weights,
            bias,
            act_func,
            last_input: Mutex::new(None),
            last_z: Mutex::new(None),
        }
    }

    pub fn get_weights(&self) -> &Matrix { &self.weights }
    pub fn get_bias(&self) -> &Matrix { &self.bias }
    pub fn set_weights(&mut self, w: Matrix) { self.weights = w; }
    pub fn set_bias(&mut self, b: Matrix) { self.bias = b; }
}

impl Layer for DenseLayer {
    fn forward(&self, in_matrix: &Matrix) -> Matrix {
        *self.last_input.lock().unwrap() = Some(in_matrix.clone());

        let w_x = (&self.weights * in_matrix).expect("Matris carpimi boyut uyusmazligi: W * X");
        let z = (&w_x + &self.bias).expect("Matris toplami boyut uyusmazligi: + b");

        *self.last_z.lock().unwrap() = Some(z.clone());

        let mut a = Matrix::new(z.rows, z.cols);
        for i in 0..z.rows {
            for j in 0..z.cols {
                let transformed_val = self.act_func.transform(z.get(i, j));
                *a.get_mut(i, j) = transformed_val;
            }
        }
        a
    }

    fn backward(&mut self, output_gradient: &Matrix, learning_rate: f64) -> Matrix {
        let z_guard = self.last_z.lock().unwrap();
        let z = z_guard.as_ref().expect("Once forward pass yapilmalidir.");
        let input_guard = self.last_input.lock().unwrap();
        let input = input_guard.as_ref().expect("Once forward pass yapilmalidir.");

        let mut sp = Matrix::new(z.rows, z.cols);
        for i in 0..z.rows {
            for j in 0..z.cols {
                *sp.get_mut(i, j) = self.act_func.derivative(z.get(i, j));
            }
        }

        let delta = output_gradient.hadamard(&sp).unwrap();
        let input_t = input.transpose();
        let weights_grad = (&delta * &input_t).unwrap();

        for i in 0..self.weights.rows {
            for j in 0..self.weights.cols {
                let current_w = self.weights.get(i, j);
                let grad_w = weights_grad.get(i, j);
                *self.weights.get_mut(i, j) = current_w - learning_rate * grad_w;
            }
        }

        for i in 0..self.bias.rows {
            let current_b = self.bias.get(i, 0);
            let grad_b = delta.get(i, 0);
            *self.bias.get_mut(i, 0) = current_b - learning_rate * grad_b;
        }

        let weights_t = self.weights.transpose();
        (&weights_t * &delta).unwrap()
    }
}
