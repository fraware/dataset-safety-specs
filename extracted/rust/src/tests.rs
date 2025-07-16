#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_guard() {
        let row = Row::new()
            .with_phi(vec!["SSN: 123-45-6789".to_string()]);
        assert!(phi_guard(&row));
    }
    #[test]
    fn test_coppa_guard() {
        let row = Row::new()
            .with_phi(vec!["SSN: 123-45-6789".to_string()]);
        assert!(coppa_guard(&row));
    }
    #[test]
    fn test_gdpr_guard() {
        let row = Row::new()
            .with_phi(vec!["SSN: 123-45-6789".to_string()]);
        assert!(gdpr_guard(&row));
    }
}