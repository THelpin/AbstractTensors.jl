# Tensor algebra benchmark suite for SymbolicTensors.jl
#
# Run from repo root:
#   julia --project -e 'using Pkg; Pkg.instantiate(); include("benchmark/benchmark.jl")'
#
# Or after activating the package env with BenchmarkTools available:
#   julia --project benchmark/benchmark.jl

using SymbolicTensors
using SymbolicTensors: term, coeff_of, terms_of
using BenchmarkTools
using Random

include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "bench_sort.jl"))

bench = setup_benchmark_geometry!()
(; g, H, F, T, components, tensor_ids, expr1, expr2, expr3) = bench

println("=========================================")
println("TENSOR ALGEBRA BENCHMARK SUITE")
println("=========================================")

# ---------------------------------------------------------
# Benchmark 1: FOIL distributivity stress test
# ---------------------------------------------------------
println("\n--- Benchmark 1: Distributivity (FOIL) ---")
b1 = @benchmark ($expr1 * $expr2) * $expr3
display(b1)
display((expr1 * expr2) * expr3)

# ---------------------------------------------------------
# Benchmark 2: commutative merge (1000 shuffled 4-factor products)
# ---------------------------------------------------------
Random.seed!(0xC0FFEE)
random_terms = map(1:1000) do i
    shuffled = shuffle(components)
    prod = shuffled[1] * shuffled[2] * shuffled[3] * shuffled[4]
    return term(prod)
end

println("\n--- Benchmark 2: Massive Commutative Merge ---")
b2 = @benchmark sum($random_terms)
display(b2)

final_result = sum(random_terms)
merged = terms_of(final_result)
println("\nFinal merged term count: ", length(merged))
if length(merged) == 1
    println("Final merged coefficient: ", coeff_of(merged[1]))
else
    @warn "Expected a single merged term; got $(length(merged))"
end

# ---------------------------------------------------------
# Benchmarks 3–5: is_canonical_less / head-order micro-benchmarks
# ---------------------------------------------------------
run_sort_benchmark!(components, tensor_ids)

println("\n=========================================")
println("Done.")
println("=========================================")
