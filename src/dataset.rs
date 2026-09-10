use crate::matrix::Matrix;
use rand::seq::SliceRandom;
use rand::thread_rng;
pub struct Dataset;

impl Dataset {
    pub fn to_one_hot(labels: &[usize], num_classes: usize) -> Matrix {
        let mut target = Matrix::new(num_classes, labels.len());
        for (col, &class_id) in labels.iter().enumerate() {
            if class_id < num_classes {
                *target.get_mut(class_id, col) = 1.0;
            }
        }
        target
    }

    pub fn train_test_split(
        x: &Matrix, 
        y: &Matrix, 
        test_ratio: f64
    ) -> (Matrix, Matrix, Matrix, Matrix) {
        let total_samples = x.cols;
        let test_count = (total_samples as f64 * test_ratio).round() as usize;
        let train_count = total_samples - test_count;

        let x_train = x.slice_cols(0, train_count);
        let y_train = y.slice_cols(0, train_count);
        let x_test = x.slice_cols(train_count, total_samples);
        let y_test = y.slice_cols(train_count, total_samples);

        (x_train, y_train, x_test, y_test)
    }
    pub fn shuffle(x: &mut Matrix, y: &mut Matrix) {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..x.cols).collect();
        indices.shuffle(&mut rng);

        let mut new_x = Matrix::new(x.rows, x.cols);
        let mut new_y = Matrix::new(y.rows, y.cols);

        for (new_col, &old_col) in indices.iter().enumerate() {
            for r in 0..x.rows { *new_x.get_mut(r, new_col) = x.get(r, old_col); }
            for r in 0..y.rows { *new_y.get_mut(r, new_col) = y.get(r, old_col); }
        }
        *x = new_x;
        *y = new_y;
    }
}
