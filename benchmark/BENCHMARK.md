# SymbolicTensors benchmark suite

Performance benchmarks for the tensor component algebra layer (`TensorComponentTerm`, `TensorComponentSum`, `TensorComponentProduct`). These are **development benchmarks**, not CI gates.

## Prerequisites

- Julia ≥ 1.10 (see root `Project.toml`)
- Package environment activated from the repo root
- **BenchmarkTools** as an optional extra (not a runtime dependency of SymbolicTensors)

BenchmarkTools is declared under `[extras]` / `[targets] bench`. After cloning:

```bash
cd SymbolicTensors
julia --project -e 'using Pkg; Pkg.instantiate()'
```

If `using BenchmarkTools` fails, add the bench extra explicitly:

```julia
using Pkg
Pkg.add(PackageSpec(name="BenchmarkTools", target="bench"))
Pkg.instantiate()
```

**Do not** add BenchmarkTools to `[deps]` unless you intend it to be a required dependency for all users.

## Running the suite

From the **repository root**:

```bash
julia --startup-file=no --project benchmark/benchmark.jl
```

`--startup-file=no` avoids a broken or slow `startup.jl` (e.g. Revise) interfering with load times.

Alternative (REPL):

```julia
using Pkg
Pkg.activate("SymbolicTensors")   # path to repo
using SymbolicTensors
using BenchmarkTools
include("benchmark/benchmark.jl")
```

The script prints four `BenchmarkTools.Trial` summaries and a short correctness check for benchmark 2.

## Layout

| File | Role |
|------|------|
| `benchmark.jl` | Entry point: FOIL, merge, calls sort micro-benchmarks |
| `setup.jl` | Defines manifold `BM_M`, metric `g`, tensors `H`/`F`/`T`, indices `a1`–`a8`, FOIL expressions |
| `bench_sort.jl` | Micro-benchmarks: `print_as`, Dict lookup, mock `BenchHead` field access |
| `BENCHMARK.md` | This document |

Setup uses isolated names (`BM_M`, etc.) and clears module registries so benchmarks do not clash with a loaded dev session.

## What each benchmark measures

### 1 — Distributivity (FOIL)

**Expression:** `(expr1 * expr2) * expr3` where each `expr*` is a sum of four components (`g`, `H`, `F`, `T` on cotangent slots).

**Exercises:** `TensorComponentSum` × sum multiplication (FOIL), `term` promotion, `_make_product`, canonical factor sort, `_merge_terms` on many new terms.

**Typical scale (Julia 1.12, one reference machine):** ~280 μs median, ~230 KiB allocated per run. GC spikes on outliers are normal for this workload.

### 2 — Massive commutative merge

**Workload:** 1000 terms, each `term(shuffled[1]*…*shuffled[4])` with `g,H,F,T` in random order (seed `0xC0FFEE`), then `sum(random_terms)`.

**Exercises:** Single [`Base.sum(::AbstractArray{<:TensorComponentTerm})`](@ref) call → one [`TensorComponentSum`](@ref) constructor → one `_merge_terms` pass (not a left-fold of 999 intermediate sums).

**Correctness check:** After the timed run, the script prints:

- `Final merged term count: 1`
- `Final merged coefficient: 1000`

If merge or canonicalization is wrong, you get multiple terms or a coefficient ≠ 1000.

**Typical scale:** Re-run after the bulk `sum` optimization; expect far fewer allocations than the old chained-fold baseline (~100k). Median time should drop accordingly on large term vectors.

### 3 — `is_canonical_less` (current: `print_as`)

**Workload:** 4000 comparisons (250 repeats × 4×4 component pairs) using the same logic as production (`print_as` string compare, then lexicographic indices).

**Typical scale:** ~18 μs median for the whole loop, **0 allocations** in the trial (strings already interned / no new strings on the hot path for these heads).

### 4 — `is_canonical_less` (Dict lookup — anti-pattern)

**Workload:** Same 4000 `TensorComponent` pairs as benchmark 3, but ordering uses:

1. `tensor ===` then index lex order, else
2. `ids[tensor]` from a `Dict{Any,Int}` built at setup.

This is what **not** to ship: Dict lookup dominates cost. On a reference run, print_as was ~18 μs vs Dict ~324 μs — **do not** use a `Dict` for canonical sort keys.

### 5 — Head order: `String` label vs `Int` field (mock `BenchHead`)

**Workload:** Same 4000 pairwise comparisons, but on a mock struct with only `registry_id::Int` and `label::String` (no `Tensor`, no indices, no Dict):

```julia
struct BenchHead
    registry_id::Int
    label::String
end
```

Two trials:

- `a.label < b.label` — stand-in for `print_as` string compare on different heads
- `a.registry_id < b.registry_id` — stand-in for `a.tensor.registry_id < b.tensor.registry_id`

**Purpose:** Estimate whether a direct `Int` field can beat short-string `print_as` **before** changing `Tensor`. Benchmark 5 isolates head comparison only; benchmarks 3–4 include index tie-breaks and real `TensorComponent` paths.

**How to read results:**

- If benchmark 5 field ≈ or faster than benchmark 5 label → field-based `registry_id` is worth implementing for sort at scale.
- If benchmark 5 field ≈ benchmark 3 but benchmark 4 is huge → Dict was the problem, not integers.
- Benchmark 5 field is usually **not** orders of magnitude faster than benchmark 3 at four heads; expect similar nanoseconds per compare. Gains matter more with many tensor heads and registration-order semantics.

After implementing real `registry_id` on `Tensor`, add a benchmark 6 calling production `is_canonical_less` or re-run benchmark 3 against the new implementation.

## Reading BenchmarkTools output

- **median** — usual metric for comparisons (less sensitive than mean to GC tails).
- **Memory estimate / allocs** — important for algebra code; regressions often show up here first.
- **GC %** — high on benchmark 1 is expected; benchmark 2 may show periodic GC from large allocations.

For A/B tests (e.g. before/after `registry_id`):

1. Same Julia version and `--project`.
2. Same machine load; close other heavy processes.
3. Run twice; compare medians.
4. Use `$` interpolation in `@benchmark` (already done in these scripts) so globals are not the bottleneck.

Quick single-shot timing:

```julia
using BenchmarkTools
@btime ($expr1 * $expr2) * $expr3
```

## Extending benchmarks

1. Add geometry or expressions in `setup.jl` (return new fields in the named tuple).
2. Add trials in `benchmark.jl`, or a new `bench_*.jl` included from `benchmark.jl`.
3. For sort-only experiments, edit `bench_sort.jl` or call `run_sort_benchmark!` from the REPL after `bench = setup_benchmark_geometry!()`.

Keep registry clears in setup so results stay reproducible.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `Malformed UUID` / `BenchmarkTools` not found | Bad or missing extra in `Project.toml` | Use UUID `6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf`; `Pkg.resolve()` |
| `UndefVarError` on load | Package failed to precompile | `Pkg.precompile()`; run `Pkg.test()` |
| Benchmark 2: coeff ≠ 1000 | Product canonicalization or merge bug | Fix `is_canonical_less` / `TensorComponentProduct` / `_merge_terms` |
| Very slow first run | JIT + precompile | Second run is representative; use `BenchmarkTools` trials, not first `@time` |

## Relation to tests

`test/runtests.jl` checks functional correctness (including product and sum algebra). Benchmarks measure performance only and are not run in `Pkg.test()`. Run both when changing `tensorComponentExpr.jl`:

```bash
julia --project -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project benchmark/benchmark.jl
```
