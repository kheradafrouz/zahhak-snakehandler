# Security Policy

## Supported versions
Only the latest release receives security fixes.

## Reporting a vulnerability
Please **do not** open a public issue. Use GitHub's private vulnerability reporting
(Security tab -> "Report a vulnerability") and include steps to reproduce and the affected version.
You can expect an acknowledgement within a few days.

## Notes on checkpoints
Checkpoints are loaded with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled code.
Still, only load checkpoint files from sources you trust.
