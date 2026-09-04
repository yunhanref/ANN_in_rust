// C++'taki IActivation arayuzunun karsiligi.
// Trait, implementasyon yapacak turlerin hangi metotlari saglamasi gerektigini belirtir.


use crate::matrix::Matrix;

pub trait Activation: Send + Sync {
    fn transform(&self, val: f64) -> f64;
}

// C++'taki BaseLayer arayuzunun karsiligi.
pub trait Layer: Send + Sync {
    // Girdiyi alip yeni bir Matrix dondurur.
    // Box<dyn Layer> kullanimi icin object-safe olmali.
    fn forward(&self, input: &Matrix) -> Matrix;
}