use crate::matrix::Matrix;

pub struct MSE;

impl MSE {
    // Ortalama Kare Hata (Loss)
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

    // Hata türevi (Geri yayılımın başlangıç noktası: dL/dPred)
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
