use crate::traits::Activation;
use crate::matrix::Matrix;

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn transform(&self, val: f64) -> f64 { 1.0 / (1.0 + (-val).exp()) }
    fn derivative(&self, val: f64) -> f64 {
        let s = self.transform(val);
        s * (1.0 - s)
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn transform(&self, val: f64) -> f64 { val.max(0.0) }
    fn derivative(&self, val: f64) -> f64 { if val > 0.0 { 1.0 } else { 0.0 } }
}

pub struct Tanh;
impl Activation for Tanh {
    fn transform(&self, val: f64) -> f64 { val.tanh() }
    fn derivative(&self, val: f64) -> f64 {
        let t = self.transform(val);
        1.0 - t * t
    }
}

pub struct Softmax;
impl Activation for Softmax {
    fn transform(&self, val: f64) -> f64 { val }
    fn derivative(&self, _val: f64) -> f64 { 1.0 }

    // Sayisal Kararli (Numerically Stable) Sutun Bazli Softmax
    fn apply(&self, z: &Matrix) -> Matrix {
        let mut a = Matrix::new(z.rows, z.cols);
        for j in 0..z.cols {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..z.rows {
                let v = z.get(i, j);
                if v > max_val { max_val = v; }
            }
            let mut sum_exp = 0.0;
            for i in 0..z.rows {
                let e = (z.get(i, j) - max_val).exp();
                *a.get_mut(i, j) = e;
                sum_exp += e;
            }
            for i in 0..z.rows {
                *a.get_mut(i, j) /= sum_exp.max(1e-9);
            }
        }
        a
    }

    // Cross-Entropy ile birlesik gradyan turevi dogrudan aktarilir
    fn apply_derivative(&self, output_grad: &Matrix, _z: &Matrix) -> Matrix {
        output_grad.clone()
    }
}