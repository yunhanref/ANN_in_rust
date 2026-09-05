use crate::matrix::Matrix;
use crate::errors::BrainError;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

pub struct DataHandler;

impl DataHandler {
    pub fn load_csv(path: &str, has_header: bool) -> Result<Matrix, BrainError> {
        let file = File::open(path).map_err(|e| BrainError::FileIoError(e.to_string()))?;
        let reader = BufReader::new(file);
        
        let mut rows: Vec<Vec<f64>> = Vec::new();
        
        for (i, line) in reader.lines().enumerate() {
            if has_header && i == 0 { continue; }
            let line = line.map_err(|e| BrainError::FileIoError(e.to_string()))?;
            if line.is_empty() { continue; }
            
            let mut row = Vec::new();
            for cell in line.split(',') {
                if let Ok(val) = cell.trim().parse::<f64>() {
                    row.push(val);
                }
            }
            if !row.is_empty() { rows.push(row); }
        }
        
        let n_rows = rows.len();
        let n_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        
        let mut result = Matrix::new(n_rows, n_cols);
        for i in 0..n_rows {
            for j in 0..n_cols {
                *result.get_mut(i, j) = *rows[i].get(j).unwrap_or(&0.0);
            }
        }
        Ok(result)
    }

    pub fn save_csv(path: &str, m: &Matrix) -> Result<(), BrainError> {
        let mut file = File::create(path).map_err(|e| BrainError::FileIoError(e.to_string()))?;
        for i in 0..m.rows {
            for j in 0..m.cols {
                write!(file, "{}", m.get(i, j)).unwrap();
                if j < m.cols - 1 { write!(file, ",").unwrap(); }
            }
            writeln!(file).unwrap();
        }
        Ok(())
    }
}
