pub fn load_input(name: &str) -> String {
    std::fs::read_to_string("inputs/".to_string() + name).unwrap()
}
