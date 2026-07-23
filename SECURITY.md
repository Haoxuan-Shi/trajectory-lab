# Security policy

## Supported versions

Security fixes target the latest release on the default branch.

## Reporting a vulnerability

Use the repository host's private security-advisory feature. Include the
affected version, minimal reproducer, impact, and suggested mitigation. Do not
submit secrets or private map/vehicle data in a public issue.

ASCII and CSV inputs are treated as untrusted and validated. CSV headers are
checked for blank or duplicate names before row mapping, and every row must
match the validated header width. Finite numeric inputs are also rejected when
coordinate conversion or geometric arithmetic would produce a non-finite or
unrepresentable intermediate. These failures use `InvalidInputError` (CLI
status `2`) rather than exposing implementation exceptions.

Callers should still impose application-appropriate file-size and resource
limits. This project is not certified for safety-critical navigation;
deployments require independent fail-safe controls, localization checks, and
platform-specific verification.
