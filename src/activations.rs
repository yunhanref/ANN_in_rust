use crate::traits::Activation;

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn transform(&self, val: f64) -> f64 {
        1.0 / (1.0 + (-val).exp())
    }
    fn derivative(&self, val: f64) -> f64 {
        let s = self.transform(val);
        s * (1.0 - s) // Sigmoid türevi: s * (1 - s)
    }
}
pub struct ReLU;

impl Activation for ReLU {
    fn transform(&self, val: f64) -> f64 {
        val.max(0.0)
    }
    fn derivative(&self, val: f64) -> f64 {
        if val > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct Tanh;

impl Activation for Tanh {
    fn transform(&self, val: f64) -> f64 {
        val.tanh()
    }
    fn derivative(&self, val: f64) -> f64 {
        let t = self.transform(val);
        1.0 - t * t // Tanh türevi: 1 - tanh^2(val)
    }
}
