pub trait AsciiByteSliceExt {
    fn trim(&self) -> &[u8];
    fn trim_start(&self) -> &[u8];
    fn trim_end(&self) -> &[u8];
    fn parse_usize(&self) -> usize;
    fn parse_isize(&self) -> isize;
}

// Trim implementations are copied from the still unstable `feature(byte_slice_trim_ascii)`.
// (github.com/rust-lang/rust/issues/94035)
impl AsciiByteSliceExt for [u8] {
    fn trim(&self) -> &[u8] {
        self.trim_start().trim_end()
    }

    fn trim_start(&self) -> &[u8] {
        let mut bytes = self;
        while let [first, rest @ ..] = bytes {
            if *first == b' ' {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    fn trim_end(&self) -> &[u8] {
        let mut bytes = self;
        while let [rest @ .., last] = bytes {
            if *last == b' ' {
                bytes = rest;
            } else {
                break;
            }
        }
        bytes
    }

    /// Result is only correct if the bytes represent a valid positive number without any additional
    /// characters!
    fn parse_usize(&self) -> usize {
        let mut ret = 0;
        for b in self {
            ret = ret * 10 + (b - b'0') as usize;
        }
        ret
    }

    /// Result is only correct if the bytes represent a valid number without any additional
    /// characters!
    fn parse_isize(&self) -> isize {
        let mut bytes = self;
        let mut negative = false;
        if self[0] == b'-' {
            bytes = &self[1..];
            negative = true;
        }
        let mut ret = 0;
        for b in bytes {
            ret = ret * 10 + (b - b'0') as isize;
        }
        if negative {
            -ret
        } else {
            ret
        }
    }
}
