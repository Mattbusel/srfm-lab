// types.rs — Shared types mirroring C++ structs
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable, Default)]
#[repr(u8)]
pub enum DType {
    #[default]
    Unknown  = 0xFF,
    Float32  = 0x01,
    Float16  = 0x02,
    Float64  = 0x03,
    Int32    = 0x04,
    Int64    = 0x05,
    Uint8    = 0x06,
    Complex64= 0x07,
}

impl DType {
    pub fn element_size(self) -> usize {
        match self {
            DType::Float32   => 4,
            DType::Float16   => 2,
            DType::Float64   => 8,
            DType::Int32     => 4,
            DType::Int64     => 8,
            DType::Uint8     => 1,
            DType::Complex64 => 8,
            DType::Unknown   => 0,
        }
    }

    pub fn numpy_str(self) -> &'static str {
        match self {
            DType::Float32   => "<f4",
            DType::Float16   => "<f2",
            DType::Float64   => "<f8",
            DType::Int32     => "<i4",
            DType::Int64     => "<i8",
            DType::Uint8     => "|u1",
            DType::Complex64 => "<c8",
            DType::Unknown   => "unknown",
        }
    }
}

/// TensorDescriptor — mirrors C++ TensorDescriptor (256 bytes)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct TensorDescriptor {
    pub dtype:         u8,
    pub ndim:          u8,
    pub _pad:          [u8; 6],
    pub shape:         [u64; 8],
    pub strides:       [u64; 8],
    pub num_elements:  u64,
    pub payload_bytes: u64,
    pub name:          [u8; 32],
}

impl TensorDescriptor {
    pub fn new_float32_1d(n: usize) -> Self {
        let mut td = TensorDescriptor::zeroed();
        td.dtype = DType::Float32 as u8;
        td.ndim  = 1;
        td.shape[0] = n as u64;
        td.strides[0] = 4;
        td.num_elements = n as u64;
        td.payload_bytes = (n * 4) as u64;
        td
    }

    pub fn new_float64_2d(rows: usize, cols: usize) -> Self {
        let mut td = TensorDescriptor::zeroed();
        td.dtype = DType::Float64 as u8;
        td.ndim  = 2;
        td.shape[0] = rows as u64;
        td.shape[1] = cols as u64;
        td.strides[0] = (cols * 8) as u64;
        td.strides[1] = 8;
        td.num_elements = (rows * cols) as u64;
        td.payload_bytes = (rows * cols * 8) as u64;
        td
    }

    pub fn dtype(&self) -> DType {
        match self.dtype {
            0x01 => DType::Float32,
            0x02 => DType::Float16,
            0x03 => DType::Float64,
            0x04 => DType::Int32,
            0x05 => DType::Int64,
            0x06 => DType::Uint8,
            0x07 => DType::Complex64,
            _    => DType::Unknown,
        }
    }
}

/// SlotHeader — matches C++ SlotHeader layout
#[derive(Pod, Zeroable, Clone, Copy)]
#[repr(C, align(64))]
pub struct SlotHeader {
    pub magic:        u64,
    pub sequence:     u64,  // atomic in C++, read with acquire fence in Rust
    pub timestamp_ns: u64,
    pub producer_id:  u64,
    pub flags:        u32,
    pub schema_ver:   u32,
    pub tensor:       TensorDescriptor,
    pub _pad:         [u8; 64],
}

impl SlotHeader {
    pub const FLAG_VALID:      u32 = 1 << 0;
    pub const FLAG_COMPRESSED: u32 = 1 << 1;
    pub const FLAG_LAST:       u32 = 1 << 2;
    pub const FLAG_HEARTBEAT:  u32 = 1 << 3;

    pub fn is_valid(&self) -> bool {
        self.flags & Self::FLAG_VALID != 0
    }
}

/// RingControl — ring metadata at base of shared memory region
#[derive(Pod, Zeroable, Clone, Copy)]
#[repr(C, align(64))]
pub struct RingControl {
    pub magic:        u64,
    pub slot_bytes:   u64,
    pub ring_cap:     u64,
    pub schema_ver:   u64,
    pub write_seq:    u64,   // atomic in C++
    pub min_read_seq: u64,   // atomic in C++
    pub channel_name: [u8; 64],
    pub _pad:         [u8; 64],
}
