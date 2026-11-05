# Functional specification: Password Reset
Requirement ID: REQ-RESET

## Business rule
A user can request a password reset link valid for 30 minutes. A reset token can be used only once. Expired or reused tokens are rejected.

## Acceptance objective
Generate positive, negative, boundary and access-control tests whenever relevant.
