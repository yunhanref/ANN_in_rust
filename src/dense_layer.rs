use crate::matrix::Matrix;
use crate::traits::{Layer, Activation};

pub struct DenseLayer {
    weights: Matrix,
    bias: Matrix,
    // C++'taki 'IActivation* actFunc' karsiligi.
    // 'dyn' anahtar kelimesi, bunun calisma zamaninda (runtime) cozulecegini belirtir.
    act_func: Box<dyn Activation>, 
}

impl DenseLayer {
    // C++'taki constructor mantigi
    pub fn new(in_dim: usize, out_dim: usize, act_func: Box<dyn Activation>) -> Self {
        let mut weights = Matrix::new(out_dim, in_dim);
        let mut bias = Matrix::new(out_dim, 1);
        
        // C++'taki std::srand ve randomize(-1.0, 1.0) mantiginin karsiligi
        weights.randomize(-1.0, 1.0);
        bias.randomize(-1.0, 1.0);

        DenseLayer {
            weights,
            bias,
            act_func,
        }
    }

    // C++ Getter metotlari
    pub fn get_weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn get_bias(&self) -> &Matrix {
        &self.bias
    }

    // C++ Setter metotlari
    pub fn set_weights(&mut self, w: Matrix) {
        self.weights = w;
    }

    pub fn set_bias(&mut self, b: Matrix) {
        self.bias = b;
    }
}

impl Layer for DenseLayer {
    fn forward(&self, in_matrix: &Matrix) -> Matrix {
        // 1. Lineer Donusum (Z = W * X + b)
        // Rust'ta operator overload'larimiz Result dondugu icin,
        // gercek hayatta Layer trait'inin de Result donmesi daha guvenlidir.
        // C++ mantigina birebir uymasi adina burada .expect() ile devam ediyoruz.
        let w_x = (&self.weights * in_matrix).expect("Matris carpimi boyut uyusmazligi: W * X");
        let z = (&w_x + &self.bias).expect("Matris toplami boyut uyusmazligi: + b");

        // 2. Aktivasyon Fonksiyonu
        let mut a = Matrix::new(z.rows, z.cols);
        for i in 0..z.rows {
            for j in 0..z.cols {
                let transformed_val = self.act_func.transform(z.get(i, j));
                *a.get_mut(i, j) = transformed_val;
            }
        }
        
        a
    }
}