---
name: Dataset Policy Issue
about: Report issues related to data policy filters, compliance, or PHI/GDPR/COPPA handling
title: "[POLICY] "
labels: ["policy", "compliance"]
assignees: ""
---

## Description

A clear description of the policy issue.

## Policy Type

- [ ] HIPAA PHI
- [ ] COPPA (Children's Online Privacy)
- [ ] GDPR (General Data Protection Regulation)
- [ ] Custom Policy
- [ ] Other

## Steps to Reproduce

1. Create dataset with...
2. Apply policy filter...
3. See error...

## Expected Behavior

What should happen?

## Actual Behavior

What actually happened?

## Sample Data (Optional)

Provide sample data that triggers the issue (anonymized):

```json
{
  "phi": ["SSN: ***-**-****"],
  "age": 25,
  "gdpr_special": []
}
```

## Severity

- [ ] Critical (Data breach risk)
- [ ] High (Compliance violation)
- [ ] Medium (Policy gap)
- [ ] Low (Minor issue)

## Additional Context

Any other context about the issue.
