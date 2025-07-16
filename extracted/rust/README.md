# Dataset Safety Guards - Rust

Auto-generated Rust guards from Lean predicates for dataset safety verification.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ds-guard = { path = "path/to/ds-guard" }
```

## Usage

```rust
use ds_guard::{Row, phi_guard, coppa_guard, gdpr_guard};

let row = Row::new()
    .with_phi(vec!["SSN: 123-45-6789".to_string()])
    .with_age(25);

if phi_guard(&row) {
    println!("PHI detected!");
}

if coppa_guard(&row) {
    println!("COPPA violation detected!");
}
```

## Generated Guards

- `phi_guard`: Detects Protected Health Information
- `coppa_guard`: Detects COPPA violations (age < 13)
- `gdpr_guard`: Detects GDPR special categories

## Building

```bash
cargo build
cargo test
```

## Publishing

```bash
cargo publish
```