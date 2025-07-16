# Dataset Safety Guards - Python

Auto-generated Python guards from Lean predicates for dataset safety verification.

## Installation

```bash
pip install ds-guard
```

## Usage

```python
from ds_guard import Row, phi_guard, coppa_guard, gdpr_guard

row = Row(
    phi=["SSN: 123-45-6789"],
    age=25,
    gdpr_special=[]
)

if phi_guard(row):
    print("PHI detected!")

if coppa_guard(row):
    print("COPPA violation detected!")
```

## Generated Guards

- `phi_guard`: Detects Protected Health Information
- `coppa_guard`: Detects COPPA violations (age < 13)
- `gdpr_guard`: Detects GDPR special categories

## Development

```bash
pip install -e .
python -m pytest tests/
```

## Publishing

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```