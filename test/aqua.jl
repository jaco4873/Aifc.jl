# Code-quality checks via Aqua.jl:
#   - method ambiguities (off — categorical-belief broadcasts produce false positives
#     once we add Distributions.jl integration; safer to enable per-PR)
#   - unbound type parameters
#   - stale or undeclared dependencies
#   - Project.toml hygiene
#   - persistent tasks introduced at module load
using Aqua
using Aifc

Aqua.test_all(Aifc; ambiguities = false)
