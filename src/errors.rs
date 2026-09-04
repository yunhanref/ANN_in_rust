use std::fmt;

#[derive(Debug)]
pub enum BrainError {
    // C++: DimensionMismatchException
    DimensionMismatch,
    // C++: FileIOException
    FileIoError(String),
    // C++: InvalidFormatException
    InvalidFormat,
}

impl std::error::Error for BrainError {}

impl fmt::Display for BrainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BrainError::DimensionMismatch => write!(f, "[Hata]: Matris boyutlari islem icin uyumsuzdur."),
            BrainError::FileIoError(msg) => write!(f, "[Hata]: Dosya acilamadi. Detay: {}", msg),
            BrainError::InvalidFormat => write!(f, "[Hata]: CSV format bozuk veya tutarsiz satir uzunlugu."),
        }
    }
}