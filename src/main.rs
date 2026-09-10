use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

mod matrix;
mod traits;
mod activations;
mod dense_layer;
mod network;
mod errors;
mod io;
mod loss;
mod dataset;

use crate::matrix::Matrix;
use crate::network::NeuralNetwork;
use crate::dense_layer::DenseLayer;
use crate::activations::{ReLU, Softmax, Sigmoid};
use crate::dataset::Dataset;
use crate::loss::CrossEntropy;
use crate::io::DataHandler;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 650.0]),
        ..Default::default()
    };
    eframe::run_native("OOP-Brain: Professional ANN", options, Box::new(|_cc| Box::new(AnnApp::default())))
}

#[derive(PartialEq)]
enum Tab { Egitim, Cikarim, Ayarlar }

struct AnnApp {
    current_tab: Tab,
    csv_path: String,
    target_col: usize,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    status_log: String,
    history: Vec<(f64, f64)>, 
    net: NeuralNetwork,
    inference_input: String,
}

impl Default for AnnApp {
    fn default() -> Self {
        let mut net = NeuralNetwork::new();
        net.push_layer(Box::new(DenseLayer::new(4, 16, Box::new(ReLU))));
        net.push_layer(Box::new(DenseLayer::new(16, 3, Box::new(Softmax))));

        Self {
            current_tab: Tab::Egitim,
            csv_path: "veri.csv".to_string(),
            target_col: 4,
            epochs: 200,
            learning_rate: 0.01,
            batch_size: 16,
            status_log: "Model hazir. CSV baglayin veya sentetik veriyi kullanin.".to_string(),
            history: Vec::new(),
            net,
            inference_input: "5.1, 3.5, 1.4, 0.2".to_string(),
        }
    }
}

impl eframe::App for AnnApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_pixels_per_point(2.0);

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.current_tab, Tab::Egitim, " Model Eğitimi (Train)");
                ui.selectable_value(&mut self.current_tab, Tab::Cikarim, " Çıkarım (Inference)");
                ui.selectable_value(&mut self.current_tab, Tab::Ayarlar, " Kalıcılık (Save/Load)");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.current_tab {
                Tab::Egitim => {
                    ui.horizontal(|ui| {
                        ui.label("CSV Yolu:");
                        ui.text_edit_singleline(&mut self.csv_path);
                        ui.label("Hedef Sütun İndeksi:");
                        ui.add(egui::DragValue::new(&mut self.target_col).speed(1));
                    });

                    ui.horizontal(|ui| {
                        ui.label("Epoch:");
                        ui.add(egui::DragValue::new(&mut self.epochs).speed(5).clamp_range(1..=5000));
                        ui.label("LR:");
                        ui.add(egui::DragValue::new(&mut self.learning_rate).speed(0.005).clamp_range(0.001..=1.0));
                        ui.label("Batch:");
                        ui.add(egui::DragValue::new(&mut self.batch_size).speed(2).clamp_range(1..=128));
                    });

                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button("Gerçek CSV ile Eğit").clicked() {
                            match DataHandler::load_csv(&self.csv_path, self.target_col, true) {
                                Ok((mut x, raw_labels)) => {
                                    let mut y = Dataset::to_one_hot(&raw_labels, 3);
                                    Dataset::shuffle(&mut x, &mut y);
                                    let (x_train, y_train, x_test, y_test) = Dataset::train_test_split(&x, &y, 0.2);

                                    self.history = self.net.fit(&x_train, &y_train, self.epochs, self.batch_size, self.learning_rate);

                                    let test_preds = self.net.run(&x_test);
                                    let test_acc = CrossEntropy::accuracy(&test_preds, &y_test);
                                    self.status_log = format!("CSV Eğitimi Bitti! Test Başarımı: %{:.2}", test_acc);
                                }
                                Err(e) => { self.status_log = format!("CSV HATA: {}", e); }
                            }
                        }
                    });

                    ui.separator();
                    if !self.history.is_empty() {
                        ui.label("Eğitim Kaybı (Loss) Grafiği:");
                        let points: PlotPoints = self.history.iter().enumerate()
                            .map(|(i, (loss, _))| [i as f64, *loss]).collect();
                        let line = Line::new(points);
                        Plot::new("loss_plot").height(200.0).show(ui, |plot_ui| plot_ui.line(line));
                    }
                }
                Tab::Cikarim => {
                    ui.heading("Eğitilmiş Model ile Yeni Veri Tahmini");
                    ui.label("Girdi Değerleri (Virgülle ayırın):");
                    ui.text_edit_singleline(&mut self.inference_input);
                    if ui.button("Tahmin Et (Predict)").clicked() {
                        let parts: Vec<f64> = self.inference_input.split(',')
                            .filter_map(|s| s.trim().parse().ok()).collect();

                        if parts.len() == 4 {
                            let mut test_m = Matrix::new(4, 1);
                            for (i, &val) in parts.iter().enumerate() { *test_m.get_mut(i, 0) = val; }

                            let out = self.net.run(&test_m);
                            let pred_class = out.argmax_cols()[0];
                            self.status_log = format!("Tahmin Edilen Sınıf: {}\nOlasılık Dağılımı:\nSınıf 0: {:.4}\nSınıf 1: {:.4}\nSınıf 2: {:.4}",
                                                      pred_class, out.get(0,0), out.get(1,0), out.get(2,0));
                        } else {
                            self.status_log = "Hata: Modele 4 adet girdi vermelisiniz.".to_string();
                        }
                    }
                }
                Tab::Ayarlar => {
                    ui.heading("Model Kalıcılığı");
                    if ui.button("Eğitilmiş Modeli Kaydet (model_weights.txt)").clicked() {
                        match DataHandler::save_model(&self.net, "model_weights.txt") {
                            Ok(_) => self.status_log = "Ağırlıklar başarıyla diske kaydedildi!".to_string(),
                            Err(e) => self.status_log = format!("Kaydetme Hatası: {}", e),
                        }
                    }
                }
            }

            ui.separator();
            ui.label("Sistem Logları:");
            egui::ScrollArea::vertical().show(ui, |ui| { ui.monospace(&self.status_log); });
        });
    }
}
