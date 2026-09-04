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
            
            if ui.button("Rapor 5.3: XOR Demo Testini Calistir").clicked() {
                // Ağın Kurulumu
                let mut xor_net = NeuralNetwork::new();
                
                let layer1 = DenseLayer::new(2, 2, Box::new(Sigmoid));
                let layer2 = DenseLayer::new(2, 1, Box::new(Sigmoid));
                
                xor_net.push_layer(Box::new(layer1));
                xor_net.push_layer(Box::new(layer2));

                // Girdi Matrisi 
                let mut xor_input = Matrix::new(2, 4);
                *xor_input.get_mut(0, 0) = 0.0; *xor_input.get_mut(1, 0) = 0.0;
                *xor_input.get_mut(0, 1) = 0.0; *xor_input.get_mut(1, 1) = 1.0;
                *xor_input.get_mut(0, 2) = 1.0; *xor_input.get_mut(1, 2) = 0.0;
                *xor_input.get_mut(0, 3) = 1.0; *xor_input.get_mut(1, 3) = 1.0;

                // İleri Besleme
                let xor_output = xor_net.run(&xor_input);
                
                self.xor_result = format!(
                    "Girdi (0,0) -> Ag Ciktisi: {:.4}\n\
                     Girdi (0,1) -> Ag Ciktisi: {:.4}\n\
                     Girdi (1,0) -> Ag Ciktisi: {:.4}\n\
                     Girdi (1,1) -> Ag Ciktisi: {:.4}\n\n\
                     [Basarili] XOR yapisi calistirildi.",
                     xor_output.get(0, 0), xor_output.get(0, 1), 
                     xor_output.get(0, 2), xor_output.get(0, 3)
                );
            }
            
            ui.add_space(10.0);
            
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label(&self.xor_result);
            });
        });
    }
}