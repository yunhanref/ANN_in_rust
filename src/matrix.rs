use rand::Rng;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub};


#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {

    pub fn slice_cols(&self, start_col: usize, end_col: usize) -> Matrix {
        let cols = end_col - start_col;
        let mut sub = Matrix::new(self.rows, cols);
        for i in 0..self.rows {
            for (new_j, j) in (start_col..end_col).enumerate() {
                *sub.get_mut(i, new_j) = self.get(i, j);
            }
        }
        sub
    }

    pub fn argmax_cols(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.cols);
        for j in 0..self.cols {
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0;
            for i in 0..self.rows {
                let val = self.get(i, j);
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }
            indices.push(max_idx);
        }
        indices
    }

    pub fn normalize_rows(&mut self) {
        for i in 0..self.rows {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..self.cols {
                let val = self.get(i, j);
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
            let diff = max_val - min_val;
            if diff > 1e-7 {
                for j in 0..self.cols {
                    let val = self.get(i, j);
                    *self.get_mut(i, j) = (val - min_val) / diff;
                }
            }
        }
    }
    pub fn new(r: usize, c: usize) -> Self {
        Matrix {
            rows: r,
            cols: c,
            data: vec![0.0; r * c],
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                *result.get_mut(j, i) = self.get(i, j);
            }
        }
        result
    }

    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimension mismatch".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] * other.data[i];
        }
        Ok(result)
    }

    pub fn randomize(&mut self, min_val: f64, max_val: f64) {
        let mut rng = rand::thread_rng();
        for val in self.data.iter_mut() {
            *val = rng.gen_range(min_val..=max_val);
        }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn mul(self, other: &Matrix) -> Self::Output {
        if self.cols != other.rows {
            return Err("Dimension mismatch".to_string());
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                *result.get_mut(i, j) = sum;
            }
        }
        Ok(result)
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, other: &Matrix) -> Self::Output {
        if self.rows == other.rows && other.cols == 1 {
            let mut result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                let val = other.get(i, 0);
                for j in 0..self.cols {
                    *result.get_mut(i, j) = self.get(i, j) + val;
                }
            }
            return Ok(result);
        }
        if self.cols == other.cols && other.rows == 1 {
            let mut result = Matrix::new(self.rows, self.cols);
            for j in 0..self.cols {
                let val = other.get(0, j);
                for i in 0..self.rows {
                    *result.get_mut(i, j) = self.get(i, j) + val;
                }
            }
            return Ok(result);
        }
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimension mismatch".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] + other.data[i];
        }
        Ok(result)
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, other: &Matrix) -> Self::Output {
        if self.rows == other.rows && other.cols == 1 {
            let mut result = Matrix::new(self.rows, self.cols);
            for i in 0..self.rows {
                let val = other.get(i, 0);
                for j in 0..self.cols {
                    *result.get_mut(i, j) = self.get(i, j) - val;
                }
            }
            return Ok(result);
        }
        if self.cols == other.cols && other.rows == 1 {
            let mut result = Matrix::new(self.rows, self.cols);
            for j in 0..self.cols {
                let val = other.get(0, j);
                for i in 0..self.rows {
                    *result.get_mut(i, j) = self.get(i, j) - val;
                }
            }
            return Ok(result);
        }
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Dimension mismatch".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] - other.data[i];
        }
        Ok(result)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, scalar: f64) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] * scalar;
        }
        result
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;
    fn add(self, scalar: f64) -> Self::Output {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..(self.rows * self.cols) {
            result.data[i] = self.data[i] + scalar;
        }
        result
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        if self.rows == other.rows && self.cols == other.cols {
            for i in 0..(self.rows * self.cols) {
                self.data[i] += other.data[i];
            }
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{}\t", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
