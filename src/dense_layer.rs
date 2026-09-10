use crate::matrix::Matrix;
use crate::traits::{Layer, Activation};
use std::sync::Mutex;

pub struct DenseLayer {
    pub weights: Matrix,
    pub bias: Matrix,
    act_func: Box<dyn Activation>,
    last_input: Mutex<Option<Matrix>>,
    last_z: Mutex<Option<Matrix>>,
}

impl DenseLayer {
    pub fn new(in_dim: usize, out_dim: usize, act_func: Box<dyn Activation>) -> Self {
        let mut weights = Matrix::new(out_dim, in_dim);
        let mut bias = Matrix::new(out_dim, 1);
        weights.randomize(-0.5, 0.5);
        bias.randomize(-0.1, 0.1);

        DenseLayer {
            weights,
            bias,
            act_func,
            last_input: Mutex::new(None),
            last_z: Mutex::new(None),
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, in_matrix: &Matrix) -> Matrix {
        *self.last_input.lock().unwrap() = Some(in_matrix.clone());

        let w_x = (&self.weights * in_matrix).expect("Matris carpimi hatasi: W * X");
        let z = (&w_x + &self.bias).expect("Matris toplami hatasi: + b");
        
        *self.last_z.lock().unwrap() = Some(z.clone());

        self.act_func.apply(&z)
    }

    fn backward(&mut self, output_gradient: &Matrix, learning_rate: f64) -> Matrix {
        let z_guard = self.last_z.lock().unwrap();
        let z = z_guard.as_ref().expect("Once forward pass yapilmalidir.");
        let input_guard = self.last_input.lock().unwrap();
        let input = input_guard.as_ref().expect("Once forward pass yapilmalidir.");

        // Aktivasyona gore gradyan duzeltmesi (Softmax icin direct pass)
        let delta = self.act_func.apply_derivative(output_gradient, z);

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
            let mut bias_sum = 0.0;
            for j in 0..delta.cols {
                bias_sum += delta.get(i, j);
            }
            let current_b = self.bias.get(i, 0);
            *self.bias.get_mut(i, 0) = current_b - learning_rate * bias_sum;
        }

        let weights_t = self.weights.transpose();
        (&weights_t * &delta).unwrap()
    }
    fn get_params(&self) -> Option<(Matrix, Matrix)> {
        Some((self.weights.clone(), self.bias.clone()))
    }

    fn set_params(&mut self, w: &Matrix, b: &Matrix) -> Result<(), String> {
        if self.weights.rows != w.rows || self.weights.cols != w.cols {
            return Err("Agirlik matrisi boyutlari uyusmuyor.".to_string());
        }
        self.weights = w.clone();
        self.bias = b.clone();
        Ok(())
    }
}