# pymdp cross-validation

This directory holds the infrastructure to cross-check
`Aifc.jl` outputs against
[pymdp](https://github.com/infer-actively/pymdp) — the Python reference
implementation of discrete active inference (Heins et al. 2022, JOSS
7(73): 4098).

## Status

**Not yet wired into CI.** Currently `Aifc.jl`
cross-validates against an independent TypeScript reference
implementation of the same T-maze (see
`test/unit/integration/test_tmaze_reference.jl`), which itself was
validated against analytical ground truth and the canonical Friston
references. pymdp adds an independent third reference, useful but
redundant in some sense, since both implementations follow the same
mathematical specifications.

## How to set up

Once, in a fresh Python venv:

```bash
cd reference/pymdp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to regenerate fixtures

After any change to either the Julia model or pymdp's API, regenerate the
reference values:

```bash
python generate_tmaze_fixtures.py
```

This produces `tmaze_fixtures.json` containing the canonical T-maze
A/B/C/D matrices and the inference outputs (per-step posterior, EFE
values, action distributions) for a fixed observation sequence.

## How to add a Julia cross-validation test

Once `tmaze_fixtures.json` exists in this directory, create
`test/golden/test_pymdp_tmaze.jl` along the lines of:

```julia
using JSON3        # or JSON.jl, your preference
using Distributions: probs

@testset "pymdp T-maze cross-validation" begin
    fixtures = JSON3.read(read(joinpath(@__DIR__, "..", "..", "reference",
                                          "pymdp", "tmaze_fixtures.json"),
                                 String))

    # Build our T-maze with the same parameters
    m = tmaze_model(cue_reliability=fixtures.cue_reliability,
                     preference=fixtures.preference)

    # Verify A and B match pymdp's encoding (modulo per-modality conventions)
    @test m.A ≈ fixtures.A atol=1e-12
    @test m.B ≈ fixtures.B atol=1e-12

    # Per-step posterior agreement on a fixed observation sequence
    alg = FixedPointIteration(num_iter=20, dF_tol=1e-12)
    qs = state_prior(m)
    for (t, obs) in enumerate(fixtures.observations)
        qs = infer_states(alg, m, qs, obs)
        @test probs(qs) ≈ fixtures.posteriors[t] atol=1e-6
    end
end
```

Then add it to `test/runtests.jl`.

## Versions pinned

- pymdp `>=0.0.7.1` (latest at writing)
- numpy, scipy as transitive deps

We deliberately don't pin a specific pymdp version — the API has been
stable since 0.0.7 and we want forward-compatibility. If pymdp evolves
in a way that affects the canonical T-maze numerically, this script
regenerates the fixtures and the Julia tests adapt by simply re-running.
