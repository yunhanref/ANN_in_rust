use eframe::egui;
// Projenin diger modullerini projene uygun sekilde cagiriyoruz:
// mod matrix; mod traits; mod activations; mod dense_layer; mod network; mod errors; mod io;

mod matrix;
mod traits;
mod activations;
mod dense_layer;
mod network;
mod errors;
mod io;
mod loss;

use crate::matrix::Matrix;
use crate::network::NeuralNetwork;
use crate::dense_layer::DenseLayer;
use crate::activations::Sigmoid;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        // Yeni eframe surumu icin pencere boyutu ayari
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "OOP-Brain Rust GUI Edition",
        options,
        Box::new(|_cc| Box::new(BrainApp::default())),
    )
}

struct BrainApp {
    xor_result: String,
}

impl Default for BrainApp {
    fn default() -> Self {
        Self {
            xor_result: "XOR Testi henuz calistirilmadi.".to_string(),
        }
    }
}

impl eframe::App for BrainApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(5.0);
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("GRUP 7: OOP-BRAIN SINIR AGI KUTUPHANESI");
            ui.separator();

            if ui.button("XOR Agini Egit (1000 Epoch)").clicked() {
                let mut xor_net = NeuralNetwork::new();

                // XOR problemi için gizli katmanlı yapı
                xor_net.push_layer(Box::new(DenseLayer::new(2, 4, Box::new(Sigmoid))));
                xor_net.push_layer(Box::new(DenseLayer::new(4, 1, Box::new(Sigmoid))));

                let mut xor_input = Matrix::new(2, 4);
                *xor_input.get_mut(0, 0) = 0.0; *xor_input.get_mut(1, 0) = 0.0;
                *xor_input.get_mut(0, 1) = 0.0; *xor_input.get_mut(1, 1) = 1.0;
                *xor_input.get_mut(0, 2) = 1.0; *xor_input.get_mut(1, 2) = 0.0;
                *xor_input.get_mut(0, 3) = 1.0; *xor_input.get_mut(1, 3) = 1.0;

                let mut xor_target = Matrix::new(1, 4);
                *xor_target.get_mut(0, 0) = 0.0;
                *xor_target.get_mut(0, 1) = 1.0;
                *xor_target.get_mut(0, 2) = 1.0;
                *xor_target.get_mut(0, 3) = 0.0;

                let mut final_loss = 0.0;
                // 1000 tur (epoch) boyunca eğit
                for _ in 0..1000 {
                    final_loss = xor_net.train_step(&xor_input, &xor_target, 0.5);
                }

                let trained_output = xor_net.run(&xor_input);

                self.xor_result = format!(
                    "Egitim Tamamlandi! Son Hata (Loss): {:.6}\n\n\
                    Tahminler:\n\
                    Girdi (0,0) -> Hedef: 0.0 | Tahmin: {:.4}\n\
                    Girdi (0,1) -> Hedef: 1.0 | Tahmin: {:.4}\n\
                    Girdi (1,0) -> Hedef: 1.0 | Tahmin: {:.4}\n\
                    Girdi (1,1) -> Hedef: 0.0 | Tahmin: {:.4}",
                    final_loss,
                    trained_output.get(0, 0), trained_output.get(0, 1),
                    trained_output.get(0, 2), trained_output.get(0, 3)
                );
            }

            ui.add_space(10.0);
            
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(&self.xor_result);
            });
        });
    }
}
