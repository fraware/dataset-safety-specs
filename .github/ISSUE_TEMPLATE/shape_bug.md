---
name: Shape Safety Bug
about: Report issues related to shape safety verification, ONNX parsing, or tensor shape consistency
title: "[SHAPE] "
labels: ["shape", "onnx", "verification"]
assignees: ""
---

## Description

A clear description of the shape safety issue.

## Model Information

- **Model Type**: [ONNX/PyTorch/TensorFlow/Custom]
- **Model Details**: [Architecture, size, framework]
- **Input Shape**: [e.g., [batch_size, sequence_length]]
- **Output Shape**: [e.g., [batch_size, sequence_length, hidden_size]]

## Steps to Reproduce

1. Load model from...
2. Run shape verification...
3. See error...

## Expected Behavior

What should happen during shape verification?

## Actual Behavior

What actually happened during shape verification?

## Error Message

```
Copy the full error message here
```

## Verification Step

- [ ] ONNX Parsing
- [ ] Shape Inference
- [ ] Lean Conversion
- [ ] Proof Generation
- [ ] End-to-End Verification
- [ ] Other

## Model File (Optional)

If possible, provide a link to the model file or describe how to obtain it.

## Severity

- [ ] Critical (Model crashes)
- [ ] High (Shape inconsistency)
- [ ] Medium (Verification failure)
- [ ] Low (Minor issue)

## Additional Context

Any other context about the issue.
