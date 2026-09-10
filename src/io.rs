use crate::matrix::Matrix;
use crate::errors::BrainError;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use crate::network::NeuralNetwork;

pub struct DataHandler;

impl DataHandler {
    pub fn load_csv(path: &str, target_col: usize, skip_header: bool) -> Result<(Matrix, Vec<usize>), String> {
        let file = File::open(path).map_err(|e| format!("Dosya acilamadi: {}", e))?;
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        let mut labels = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if skip_header && i == 0 { continue; }
            let l = line.map_err(|_| "Satir okuma hatasi".to_string())?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.is_empty() || l.trim().is_empty() { continue; }

            let mut row = Vec::new();
            for (col_idx, part) in parts.iter().enumerate() {
                let val = part.trim().parse::<f64>().unwrap_or(0.0);
                if col_idx == target_col {
                    labels.push(val as usize);
                } else {
                    row.push(val);
                }
            }
            data.push(row);
        }

        if data.is_empty() { return Err("Veri bulunamadi".to_string()); }
        let rows = data[0].len();
        let cols = data.len();
        let mut x = Matrix::new(rows, cols);

        for (col_idx, row_vec) in data.iter().enumerate() {
            for (row_idx, &val) in row_vec.iter().enumerate() {
                *x.get_mut(row_idx, col_idx) = val;
            }
        }
        Ok((x, labels))
    }

    pub fn save_model(net: &NeuralNetwork, path: &str) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        for (i, layer) in net.pipeline.iter().enumerate() {
            if let Some((w, b)) = layer.get_params() {
                writeln!(file, "LAYER:{}", i).map_err(|e| e.to_string())?;
                // Agirliklari bas
                for r in 0..w.rows {
                    let row_strs: Vec<String> = (0..w.cols).map(|c| w.get(r, c).to_string()).collect();
                    writeln!(file, "W:{}", row_strs.join(",")).map_err(|e| e.to_string())?;
                }
                // Biaslari bas
                for r in 0..b.rows {
                    writeln!(file, "B:{}", b.get(r, 0)).map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
}
