use rand::Rng;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Sub}; // Temizlenmis import listesi

// C++'taki pointer tabanli "data" yerine Rust'in guvenli vektoru kullaniliyor.
// Copy trait'i buyuk veriler icin implemente edilmez, bunun yerine Clone kullanilir.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    // 1. Kurucu (Constructor) - Sifirlarla doldurur (C++: Matrix(int r, int c))
    pub fn new(r: usize, c: usize) -> Self {
        Matrix {
            rows: r,
            cols: c,
            data: vec![0.0; r * c],
        }
    }
    // NOT: Yikici (Destructor), Kopya/Tasima Kuruculari ve Atama operatorleri
    // Rust derleyicisi tarafindan otomatik ve guvenli bir sekilde yonetilir.

    // 7. Eleman Erisimi (at ve operator())
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }

    // 13. Temel Matris Islemleri
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

// 8. Matris Carpimi (operator*)
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

// 9. Matris Toplamasi (operator+) - Broadcasting Destegi Eklenmis Hali
impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, other: &Matrix) -> Self::Output {
        // Kolon bazli broadcasting
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
        // Satir bazli broadcasting
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
        // Normal eleman bazli toplama
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

// 10. Matris Cikarmasi (operator-) - Broadcasting Destegi Eklenmis Hali
impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, other: &Matrix) -> Self::Output {
        // Kolon bazli broadcasting
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
        // Satir bazli broadcasting
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
        // Normal eleman bazli cikarma
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

// 11. Skaler Islemler
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

// 12. Bilesik Atama Operatorleri
impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, other: &Matrix) {
        if self.rows == other.rows && self.cols == other.cols {
            for i in 0..(self.rows * self.cols) {
                self.data[i] += other.data[i];
            }
        }
        // Karmaşık boyut uyumsuzlukları (broadcasting hataları) 
        // için Rust panikleri tetikleyebilir veya Result donduren yontemler tercih edilebilir.
    }
}

// 14. Yazdirma Yardimcilari (operator<<)
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