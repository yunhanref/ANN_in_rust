use crate::matrix::Matrix;

pub struct MSE;

impl MSE {
    pub fn forward(predicted: &Matrix, target: &Matrix) -> f64 {
        let mut sum = 0.0;
        let n = (predicted.rows * predicted.cols) as f64;
        for i in 0..predicted.rows {
            for j in 0..predicted.cols {
                let diff = predicted.get(i, j) - target.get(i, j);
                sum += diff * diff;
            }
        }
        sum / n
    }

    pub fn derivative(predicted: &Matrix, target: &Matrix) -> Matrix {
        let mut result = Matrix::new(predicted.rows, predicted.cols);
        let n = (predicted.rows * predicted.cols) as f64;
        for i in 0..predicted.rows {
            for j in 0..predicted.cols {
                let diff = predicted.get(i, j) - target.get(i, j);
                *result.get_mut(i, j) = (2.0 / n) * diff;
            }
        }
        result
    }
}

pub struct CrossEntropy;

impl CrossEntropy {
    pub fn forward(predicted: &Matrix, target: &Matrix) -> f64 {
        let mut total_loss = 0.0;
        let eps = 1e-9;
        for i in 0..predicted.rows {
            for j in 0..predicted.cols {
                let p = predicted.get(i, j).clamp(eps, 1.0 - eps);
                let y = target.get(i, j);
                total_loss -= y * p.ln();
            }
        }
        total_loss / (predicted.cols as f64)
    }

    pub fn derivative(predicted: &Matrix, target: &Matrix) -> Matrix {
        let mut grad = Matrix::new(predicted.rows, predicted.cols);
        let n = predicted.cols as f64;
        for i in 0..predicted.rows {
            for j in 0..predicted.cols {
                let diff = predicted.get(i, j) - target.get(i, j);
                *grad.get_mut(i, j) = diff / n;
            }
        }
        grad
    }

    pub fn accuracy(predicted: &Matrix, target: &Matrix) -> f64 {
        let pred_classes = predicted.argmax_cols();
        let target_classes = target.argmax_cols();
        let mut correct = 0;
        for (p, t) in pred_classes.iter().zip(target_classes.iter()) {
            if p == t { correct += 1; }
        }
        (correct as f64 / pred_classes.len() as f64) * 100.0
    }
}
