/// Result is only correct if bytes represents a valid positive number without any additional
/// characters!
pub fn parse_u32_from_bytes(bytes: &[u8]) -> usize {
    let mut ret = 0;
    for b in bytes {
        ret = ret * 10 + (b - b'0') as usize;
    }
    ret
}
