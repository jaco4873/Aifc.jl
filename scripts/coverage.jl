# Run the test suite under code-coverage tracking and print a per-file
# summary. Local equivalent of CI's Codecov upload.
#
# Run from the package root:
#     julia --project=. scripts/coverage.jl
#
# Note: this spawns a separate Julia process for the coverage processing
# step in a temporary environment, so `Coverage.jl` does not pollute the
# main package's Project.toml.

const PKG_ROOT = abspath(joinpath(@__DIR__, ".."))
const COV_DIR  = joinpath(PKG_ROOT, ".coverage")
isdir(COV_DIR) && rm(COV_DIR; force=true, recursive=true)
mkpath(COV_DIR)

# Step 1 — run the test suite with coverage tracking
println("Running test suite under coverage tracking...")
run(Cmd(`julia --project=$PKG_ROOT --code-coverage=user -e "import Pkg; Pkg.test(coverage=true)"`))

# Step 2 — process coverage data in a temp environment
println()
println("Processing coverage data...")
process_script = """
import Pkg
Pkg.activate(temp = true)
Pkg.add(Pkg.PackageSpec(name = "Coverage", version = "1"))
using Coverage

cd($(repr(PKG_ROOT))) do
    coverage = process_folder("src")
    covered, total = get_summary(coverage)
    pct = total == 0 ? NaN : 100 * covered / total

    println()
    println("="^70)
    println("Coverage summary")
    println("="^70)
    println("  Lines covered: \$covered / \$total (\$(round(pct; digits=2))%)")
    println()

    println(rpad("File", 60), rpad("Cov%", 10), "Lines")
    println("-"^80)

    rows = map(coverage) do f
        cov, tot = get_summary([f])
        p = tot == 0 ? NaN : 100 * cov / tot
        (file=relpath(f.filename, $(repr(PKG_ROOT))), pct=p, covered=cov, total=tot)
    end
    sort!(rows; by = r -> isnan(r.pct) ? 100.0 : r.pct)
    for r in rows
        println(rpad(r.file, 60),
                rpad(isnan(r.pct) ? "-" : "\$(round(r.pct; digits=1))%", 10),
                "\$(r.covered) / \$(r.total)")
    end

    LCOV.writefile(joinpath($(repr(COV_DIR)), "lcov.info"), coverage)
    clean_folder("src")
    clean_folder("test")

    println()
    println("LCOV report at \$(joinpath($(repr(COV_DIR)), "lcov.info"))")
end
"""

run(Cmd(`julia -e $process_script`))
