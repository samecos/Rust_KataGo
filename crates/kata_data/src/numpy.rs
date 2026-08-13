//! NumPy `.npy` buffer writing and `.npz` zip writing.
//!
//! Corresponds to `cpp/dataio/numpywrite.h` and `cpp/dataio/numpywrite.cpp`.
//!
//! This module intentionally does not try to read `.npy`/`.npz` files; it only
//! produces the format used by KataGo's training data export.

use kata_core::global::IOError;
use std::fs::File;
use std::io::Write;
use std::mem;

/// Total header bytes in the `.npy` format produced here.
///
/// This includes the 6-byte magic, the 2-byte version, the 2-byte header-length
/// word, and the 246-byte ASCII dictionary.
pub const TOTAL_HEADER_BYTES: usize = 256;

/// Marker trait for types that can be written as numpy array elements.
///
/// # Safety
/// Implementors must be plain-old-data with no padding or interior mutability
/// so that a slice can be safely reinterpreted as a byte slice.
pub unsafe trait NumpyElement: Copy + Default {}

unsafe impl NumpyElement for f32 {}
unsafe impl NumpyElement for f64 {}
unsafe impl NumpyElement for bool {}
unsafe impl NumpyElement for u8 {}
unsafe impl NumpyElement for u16 {}
unsafe impl NumpyElement for u32 {}
unsafe impl NumpyElement for u64 {}
unsafe impl NumpyElement for i8 {}
unsafe impl NumpyElement for i16 {}
unsafe impl NumpyElement for i32 {}
unsafe impl NumpyElement for i64 {}

fn dtype_for<T: NumpyElement>() -> &'static str {
    let le = cfg!(target_endian = "little");
    match mem::size_of::<T>() {
        1 => match std::any::type_name::<T>() {
            "bool" => "|b1",
            _ if std::any::type_name::<T>() == "u8" => "|u1",
            _ if std::any::type_name::<T>() == "i8" => "|i1",
            _ => unreachable!(),
        },
        _ => {
            let kind = match std::any::type_name::<T>() {
                "f32" => 'f',
                "f64" => 'f',
                "u16" | "u32" | "u64" => 'u',
                "i16" | "i32" | "i64" => 'i',
                _ => unreachable!(),
            };
            let endian = if le { '<' } else { '>' };
            // Sized format built at compile time via a const helper.
            dtype_sized(endian, kind, mem::size_of::<T>())
        }
    }
}

const fn dtype_sized(endian: char, kind: char, size: usize) -> &'static str {
    match (endian, kind, size) {
        ('<', 'f', 4) => "<f4",
        ('<', 'f', 8) => "<f8",
        ('<', 'u', 2) => "<u2",
        ('<', 'u', 4) => "<u4",
        ('<', 'u', 8) => "<u8",
        ('<', 'i', 2) => "<i2",
        ('<', 'i', 4) => "<i4",
        ('<', 'i', 8) => "<i8",
        ('>', 'f', 4) => ">f4",
        ('>', 'f', 8) => ">f8",
        ('>', 'u', 2) => ">u2",
        ('>', 'u', 4) => ">u4",
        ('>', 'u', 8) => ">u8",
        ('>', 'i', 2) => ">i2",
        ('>', 'i', 4) => ">i4",
        ('>', 'i', 8) => ">i8",
        _ => "",
    }
}

/// Reinterpret a slice of numpy elements as a byte slice.
///
/// # Safety
/// `T` must implement `NumpyElement`, which guarantees a POD layout.
fn element_slice_as_bytes<T: NumpyElement>(slice: &[T]) -> &[u8] {
    // SAFETY: `T: NumpyElement` is a plain-old-data type with no padding.
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), mem::size_of_val(slice)) }
}

/// A pre-allocated buffer for writing a `.npy` file.
pub struct NumpyBuffer<T: NumpyElement> {
    /// The full backing data, including the reserved header region at the start.
    pub data: Vec<T>,
    /// Requested shape. The leading dimension may be partially filled; call
    /// `prepare_header_with_num_rows` with the actual number of rows written.
    pub shape: Vec<i64>,
    dtype: &'static str,
    shape_start_byte: usize,
}

impl<T: NumpyElement> NumpyBuffer<T> {
    /// Create a buffer with the given shape.
    ///
    /// The buffer is initialized to zero/false. All dimensions must be
    /// non-negative and the total number of elements must not overflow.
    pub fn new(shape: Vec<i64>) -> Result<Self, IOError> {
        if shape.is_empty() {
            return Err(IOError("NumpyBuffer shape must be non-empty".to_string()));
        }
        let mut data_len: i64 = 1;
        for &dim in &shape {
            if dim < 0 {
                return Err(IOError(
                    "NumpyBuffer shape dimensions must be non-negative".to_string(),
                ));
            }
            data_len = data_len
                .checked_mul(dim)
                .ok_or_else(|| IOError("NumpyBuffer shape overflows".to_string()))?;
        }
        let data_len = data_len as usize;

        let size_of_t = mem::size_of::<T>();
        if TOTAL_HEADER_BYTES % size_of_t != 0 {
            return Err(IOError(format!(
                "NumpyBuffer header size {} is not a multiple of element size {}",
                TOTAL_HEADER_BYTES, size_of_t
            )));
        }

        let data = vec![T::default(); data_len];
        let dtype = dtype_for::<T>();

        let first_half = format!("{{'descr':'{}','fortran_order':False,'shape':(", dtype);
        if first_half.len() > TOTAL_HEADER_BYTES - 40 {
            return Err(IOError(format!(
                "Numpy header dict is too long for dtype {}",
                dtype
            )));
        }
        let shape_start_byte = first_half.len() + 10;

        Ok(Self {
            data,
            shape,
            dtype,
            shape_start_byte,
        })
    }

    /// Number of elements that would be written if the leading dimension has
    /// `num_writeable_rows` rows.
    pub fn get_actual_data_len(&self, num_writeable_rows: i64) -> i64 {
        let mut actual = 1i64;
        for (i, &dim) in self.shape.iter().enumerate() {
            let x = if i == 0 { num_writeable_rows } else { dim };
            actual *= x;
        }
        actual
    }

    /// Write the numpy header using the actual number of rows written and
    /// return the bytes that should be written to a file.
    ///
    /// The returned vector contains the 256-byte header followed by exactly
    /// `num_writeable_rows * product(shape[1..]) * sizeof(T)` data bytes.
    pub fn prepare_header_with_num_rows(
        &self,
        num_writeable_rows: i64,
    ) -> Result<Vec<u8>, IOError> {
        if num_writeable_rows < 0 || num_writeable_rows > self.shape[0] {
            return Err(IOError(format!(
                "num_writeable_rows {} out of range [0, {}]",
                num_writeable_rows, self.shape[0]
            )));
        }

        let actual_data_len = self.get_actual_data_len(num_writeable_rows) as usize;
        let mut out =
            Vec::with_capacity(TOTAL_HEADER_BYTES + actual_data_len * mem::size_of::<T>());
        out.resize(TOTAL_HEADER_BYTES, 0);

        // Magic and version.
        let header = &mut out[..TOTAL_HEADER_BYTES];
        header[0] = 0x93;
        header[1..6].copy_from_slice(b"NUMPY");
        header[6] = 0x01;
        header[7] = 0x00;
        let header_dict_len = TOTAL_HEADER_BYTES - 10;
        header[8] = (header_dict_len & 0xFF) as u8;
        header[9] = ((header_dict_len >> 8) & 0xFF) as u8;

        // Dictionary prefix.
        let prefix = format!("{{'descr':'{}','fortran_order':False,'shape':(", self.dtype);
        let prefix_bytes = prefix.as_bytes();
        header[10..10 + prefix_bytes.len()].copy_from_slice(prefix_bytes);

        // Shape.
        let mut idx = self.shape_start_byte;
        for (i, &dim) in self.shape.iter().enumerate() {
            if i > 0 {
                if idx >= TOTAL_HEADER_BYTES - 1 {
                    return Err(IOError("Numpy header is too long".to_string()));
                }
                header[idx] = b',';
                idx += 1;
            }
            let x = if i == 0 { num_writeable_rows } else { dim };
            let digits = format!("{}", x);
            let digits_bytes = digits.as_bytes();
            if idx + digits_bytes.len() >= TOTAL_HEADER_BYTES - 1 {
                return Err(IOError("Numpy header is too long".to_string()));
            }
            header[idx..idx + digits_bytes.len()].copy_from_slice(digits_bytes);
            idx += digits_bytes.len();
        }

        if idx >= TOTAL_HEADER_BYTES - 1 {
            return Err(IOError("Numpy header is too long".to_string()));
        }
        header[idx] = b')';
        idx += 1;
        if idx >= TOTAL_HEADER_BYTES - 1 {
            return Err(IOError("Numpy header is too long".to_string()));
        }
        header[idx] = b'}';
        idx += 1;

        // Pad with spaces and terminate with newline.
        while idx < TOTAL_HEADER_BYTES - 1 {
            header[idx] = b' ';
            idx += 1;
        }
        header[TOTAL_HEADER_BYTES - 1] = b'\n';

        // Append data bytes for the actual number of rows.
        let data_bytes = element_slice_as_bytes(&self.data[..actual_data_len]);
        out.extend_from_slice(data_bytes);

        Ok(out)
    }
}

/// A simple `.npz` writer.
pub struct ZipFile {
    file_name: String,
    writer: Option<zip::ZipWriter<File>>,
}

impl ZipFile {
    /// Create (or truncate) a zip file at the given path.
    pub fn new(file_name: &str) -> Result<Self, IOError> {
        let file = File::create(file_name)
            .map_err(|e| IOError(format!("Could not create zip file {}: {}", file_name, e)))?;
        let writer = zip::ZipWriter::new(file);
        Ok(Self {
            file_name: file_name.to_string(),
            writer: Some(writer),
        })
    }

    /// Write a named buffer into the zip archive.
    pub fn write_buffer(&mut self, name_within_zip: &str, data: &[u8]) -> Result<(), IOError> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| IOError("Zip file already closed".to_string()))?;
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        writer.start_file(name_within_zip, options).map_err(|e| {
            IOError(format!(
                "Could not start zip entry {}: {}",
                name_within_zip, e
            ))
        })?;
        writer.write_all(data).map_err(|e| {
            IOError(format!(
                "Could not write zip entry {}: {}",
                name_within_zip, e
            ))
        })?;
        Ok(())
    }

    /// Finish writing and close the archive.
    pub fn close(&mut self) -> Result<(), IOError> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| IOError("Zip file already closed".to_string()))?;
        writer.finish().map_err(|e| {
            IOError(format!(
                "Could not close zip file {}: {}",
                self.file_name, e
            ))
        })?;
        Ok(())
    }
}

impl Drop for ZipFile {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            let _ = writer.finish();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Read;

    #[test]
    fn test_numpy_buffer_header() {
        let buf = NumpyBuffer::<f32>::new(vec![4, 3]).unwrap();
        let bytes = buf.prepare_header_with_num_rows(2).unwrap();

        assert_eq!(
            bytes.len(),
            TOTAL_HEADER_BYTES + 2 * 3 * mem::size_of::<f32>()
        );
        assert_eq!(&bytes[0..6], &[0x93, b'N', b'U', b'M', b'P', b'Y']);
        assert_eq!(bytes[6], 0x01);
        assert_eq!(bytes[7], 0x00);
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!(header_len, TOTAL_HEADER_BYTES - 10);

        let header_str = String::from_utf8_lossy(&bytes[10..TOTAL_HEADER_BYTES]);
        assert!(header_str.contains("'descr':'<f4'"));
        assert!(header_str.contains("'shape':(2,3)"));
        assert!(header_str.ends_with("\n"));
    }

    #[test]
    fn test_numpy_buffer_roundtrip() {
        let mut buf = NumpyBuffer::<i32>::new(vec![2, 3]).unwrap();
        for (i, v) in [1, 2, 3, 4, 5, 6].iter().enumerate() {
            buf.data[i] = *v;
        }
        let bytes = buf.prepare_header_with_num_rows(2).unwrap();
        let data_start = TOTAL_HEADER_BYTES;
        let data_bytes = &bytes[data_start..];
        assert_eq!(data_bytes.len(), 6 * mem::size_of::<i32>());
        for (i, &v) in [1i32, 2, 3, 4, 5, 6].iter().enumerate() {
            let offset = i * mem::size_of::<i32>();
            let stored = i32::from_le_bytes([
                data_bytes[offset],
                data_bytes[offset + 1],
                data_bytes[offset + 2],
                data_bytes[offset + 3],
            ]);
            assert_eq!(stored, v);
        }
    }

    #[test]
    fn test_numpy_buffer_bool() {
        let mut buf = NumpyBuffer::<bool>::new(vec![2]).unwrap();
        buf.data[0] = true;
        buf.data[1] = false;
        let bytes = buf.prepare_header_with_num_rows(2).unwrap();
        let header_str = String::from_utf8_lossy(&bytes[10..TOTAL_HEADER_BYTES]);
        assert!(header_str.contains("'descr':'|b1'"));
        assert_eq!(bytes[TOTAL_HEADER_BYTES], 1);
        assert_eq!(bytes[TOTAL_HEADER_BYTES + 1], 0);
    }

    #[test]
    fn test_zip_file_write() {
        let tmp =
            std::env::temp_dir().join(format!("katago_numpy_test_{}.npz", std::process::id()));
        let _ = fs::remove_file(&tmp);

        {
            let mut zip = ZipFile::new(tmp.to_str().unwrap()).unwrap();
            zip.write_buffer("hello.txt", b"world").unwrap();
            zip.write_buffer("empty.txt", b"").unwrap();
            zip.close().unwrap();
        }

        assert!(tmp.exists());
        let file = File::open(&tmp).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        assert_eq!(archive.len(), 2);

        let mut contents = String::new();
        archive
            .by_name("hello.txt")
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "world");

        fs::remove_file(&tmp).unwrap();
    }
}
