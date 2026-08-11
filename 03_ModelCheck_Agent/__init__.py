"""03_ModelCheck_Agent - CAE Model & Include File Validation.

Automated checking of CAE simulation models (LS-DYNA, ANSA) by comparing
include files against reference includes. Validates model structure,
connections, materials, and boundary conditions against company-standard
reference templates to catch deviations before HPC submission.

Key capabilities:
- Compare model includes vs. reference include library
- Detect missing, modified, or outdated includes
- Validate include hierarchy and dependencies
- YAML-driven checklists (Euro NCAP 2024, company standards)
- Generate deviation reports with diff highlights
"""
