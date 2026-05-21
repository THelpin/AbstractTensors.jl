# Contributing to SymbolicTensors.jl

SymbolicTensors.jl is a symbolic tensor calculus package for theoretical physics
and differential geometry, inspired by the Mathematica package [xAct bundle](https://xact.es/index.html) and in particular [xTensor] (https://xact.es/xTensor/index.html). It provides the core infrastructure for abstract index
notation, manifolds, vector bundles, tensor definitions, connection definition, metric definition. It is the foundation layer of a planned suite of packages that will include irreducible tensor decompositions via the Brauer algebra and physics-facing tools for variational calculus, perturbations theory and explicit tensor component calculations.

If you are considering a contribution, please read this document first.

---

1. [Prerequisites](#prerequisites)
2. [Architecture Overview](#architecture-overview)
3. [Dev Workflow](#dev-workflow)
4. [Code Style Conventions](#code-style-conventions)
5. [Commit Message Conventions](#commit-message-conventions)
6. [How to Report a Bug](#how-to-report-a-bug)
7. [How to Request a Feature](#how-to-request-a-feature)
8. [How to Contribute Code](#how-to-contribute-code)
9. [Running CI Locally](#running-ci-locally)
10. [Code of Conduct](#code-of-conduct)

---

<a id="prerequisites"></a>

## Prerequisites

- Julia ≥ 1.10 (1.12 recommended)
- Git
- A working Julia environment with `Pkg` available
- Optional but strongly recommended: `Revise.jl` for development iteration
- Optional but strongly recommended: `IJulia.jl` for Jupyter notebooks

No system-level dependencies are required beyond a standard Julia installation.

---

## Architecture Overview

SymbolicTensors.jl is the **core layer** of a planned suite. Understanding this
layering is essential before contributing.

**Current package — SymbolicTensors.jl**

Provides the symbolic infrastructure:

- Manifold and vector bundle definitions
- Abstract index notations
- Coordinate and local frame bases (`Basis.category` `:coordinate` or `:frame`)
- Tensor definitions with slot structure and symmetry groups
- Connection definitions with associated covariant derivatives and curvature tensors
- Metric tensors and index raising/lowering infrastructure, Levi-civita connection and associated curvature tensors

**Planned packages in the suite**

- `Brauer.jl` — standalone implementation of the Brauer algebra, symmetric group
  algebra, Young seminormal idempotents, and projection operators for irreducible
  decomposition of tensors under GL(d,R) and O(1,d-1). This is separable from the
  geometry layer and may be independently useful to mathematicians.
- A physics-facing package: Lagrangians, field equations,
  variational calculus, irreducible decompositions of the Riemann and distortion
  tensors.
- A component-based package for concrete computations:
  explicit metric components, solutions, numerical evaluation.

---

## Dev Workflow

### 1. SymbolicTensors.jl launching usage

#### a. Julia REPL

From the package root (`SymbolicTensors/`):

```bash
julia
```

Then inside the REPL:

```julia
using SymbolicTensors
```

To get a fresh session after editing source files, restart Julia and re-run
`using SymbolicTensors`. For faster iteration during development, use
[Revise.jl](https://github.com/timholy/Revise.jl):

```julia
using Pkg; Pkg.add("Revise")   # install once
```

```bash
julia
```

```julia
using Revise
using SymbolicTensors
# source edits are now picked up automatically without restarting
```

#### b. Jupyter notebook

**Prerequisites — install the Julia kernel once**

```julia
using Pkg
Pkg.add("IJulia")
using IJulia
installkernel("Julia 1.12")
```

**Launch Jupyter from the package root:**

```bash
julia
```

```julia
using IJulia
notebook(dir=pwd())
```

Detached mode:

```julia
notebook(dir=pwd(), detached=true)
```

Directly from terminal:

```bash
julia --project=. -e 'using IJulia; notebook(dir=pwd())'
```

Notebooks live in `SymbolicTensors/notebooks/`.

---

### 2. Tests

Unit tests live in `test/runtests.jl`. All commands assume the package root as
working directory.

**One-shot from the terminal:**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**First time on a fresh clone:**

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.develop(path="."); Pkg.test()'
```

**From the Julia REPL:**

```bash
julia --project=.
```

```julia
] activate .
] instantiate    # first time only
] test
```

**Quick re-run while editing:**

```julia
using Pkg; Pkg.activate(".")
using SymbolicTensors
include("test/runtests.jl")
```

Prefer `Pkg.test()` when you want a clean run that matches CI.

---

### 3. Documentation

**Prerequisites — run once:**

```julia
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

If `SymbolicTensors` is not yet registered:

```julia
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path="."))'
```

**Build:**

```bash
julia --project=docs docs/make.jl
```

Generated site lands in `docs/build/`.

**Serve locally:**

```julia
julia -e 'using Pkg; Pkg.add("LiveServer")'  # install once
julia -e 'using LiveServer; serve(dir="docs/build")'
```

Open [http://localhost:8000](http://localhost:8000).

---

### 4. Doctests

Doctests run in the `docs` environment, not via `Pkg.test()`.

**Prerequisites — run once:**

```julia
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
```

**Launch tests:**

```bash
julia --project=docs --color=yes -e '
using Documenter: DocMeta, doctest
using SymbolicTensors
DocMeta.setdocmeta!(SymbolicTensors, :DocTestSetup, :(using SymbolicTensors); recursive=true)
doctest(SymbolicTensors)
'
```

---

## Frames and basis expansion

Each vbundle may register two basis categories in `_BASES`:

- **`:coordinate`** — coordinate-induced bases (`∂`, `dx`), indexed with
  coordinate indices (`a1`, `-a1`).
- **`:frame`** — arbitrary local frames (`e`, `θ`), indexed with basis indices
  (`A1`, `-A1`).

`basis_expansion` is **style-based only** (do not pass a `Basis` object):

```julia
basis_expansion(T)              # default: Coordinate style
basis_expansion(T, Coordinate)  # per slot: :coordinate, else :frame
basis_expansion(T, Frame)       # per slot: :frame only
```

Expansion builds canonical components from the tensor schema at `@def_tensor` time.
There is no `basis_expansion(::TensorExpression)` or `basis_expansion(::Tensor, ::Basis)`.

---

## Code Style Conventions

**Formatting**

- 4-space indentation. No tabs.
- Maximum line length 92 characters.
- One blank line between top-level definitions. Two blank lines between major
  sections in a file.

**Naming**

- Types and structs: `UpperCamelCase` — `Tensor`, `SlotSymmetry`, `TensorIndex`
- Functions and variables: `snake_case` — `rank_of`, `is_tensor`, `def_tensor`
- Macros: `@snake_case` — `@def_tensor`, `@def_manifold`
- Constants and module-level registries: `_UPPER_SNAKE_CASE` — `_TENSORS`,
  `_MANIFOLDS`
- Boolean predicates should start with `is_` — `is_tensor`, `is_up`, `is_down`

**Docstrings**

- Every exported function, macro, type, and constant must have a docstring.
- Docstrings follow the Julia convention: summary line, blank line, extended
  description, fields section for structs, examples section where appropriate.
- Internal functions marked with the `!!! warning "Internal"` admonition are
  documented but not exported.

**Example blocks in `src/*.jl` (tilde fences)**

Docstring examples use **tilde fences**, not triple backticks:

```julia
"""
    my_fn(x) -> Bool

Description here.

#### Examples

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
my_fn(a1)
~~~
"""
```

Conventions:

- Opening fence: `~~~julia` on its own line; closing fence: `~~~` on its own line.
- Prefer a `#### Examples` heading (see `src/manifolds.jl`, `src/indices.jl`).
- Leave a blank line before `#### Examples` and before `~~~julia`.
- Use `@ref` for cross-links to other documented symbols.

**Why tildes?** Julia’s Markdown parser (and Documenter, which uses it) accepts
both backtick and tilde fenced code blocks. This project uses `~~~julia` in
`src/` so example blocks do not rely on nested ` ``` ` inside docstrings, and
so AI/chat paste suggestions are less likely to corrupt files when fences are
copied.

**Not the same as `docs/src/*.md`:** manual pages use Documenter directives
(e.g. ` ```@docs `) with backticks — that syntax is unchanged. Only **docstring
examples inside `src/*.jl`** use `~~~julia`.

**Type annotations**

- Function arguments should carry type annotations where the type is meaningful
  for dispatch or documentation purposes.
- Avoid over-specifying types. Prefer `Union{Symbol, Nothing}` over forcing
  everything through a single concrete type.

**Error messages**

- Error messages must name the macro or function that threw, state what was
  expected, and suggest a fix. Example:
  `"@def_tensor: manifold M is not registered. Call @def_manifold M first."`
- Warnings via `@warn` follow the same convention.

**Tests**

- Every new exported function requires at least one test.
- Tests are grouped in `@testset` blocks named after the feature being tested.
- Each `@testset` that modifies module-level registries must call
  `_clear_all_registries!()` at the start.

---

## How to Report a Bug

Open a GitHub issue with the following information:

1. **Julia version** — output of `julia --version`
2. **OS** — Linux, macOS, Windows and version
3. **SymbolicTensors.jl version** — output of `] status SymbolicTensors`
4. **Minimal reproducible example** — the smallest possible code that triggers
   the bug. If it requires more than 10 lines to reproduce, something is probably
   missing from the example.
5. **Expected behavior** — what you expected to happen
6. **Actual behavior** — what actually happened, including the full error message
   and stack trace

Do not open a bug report for a feature request. Use the feature request template
instead.

---

## How to Request a Feature

Open a GitHub issue with the label `enhancement` and include:

1. **What you want to do** — describe the computation or workflow you are trying
   to achieve, not the implementation you have in mind
2. **Why it belongs in this layer** — explain why this feature belongs in
   `SymbolicTensors.jl` specifically rather than in a higher layer of the suite.
   If you are unsure, say so — that is a valid answer and opens a useful
   discussion.
3. **What you have tried** — have you attempted a workaround? What were its
   limitations?

**Before opening a feature request**, check that:

- The feature does not belong in `Brauer.jl` or the physics-facing layer
- You have read the existing source code well enough to understand where the
  feature would sit

Large feature requests with no prior discussion will be closed. Open an issue
first, discuss the design, then submit a PR.

---

## How to Contribute Code

1. **Open an issue first** for anything beyond a trivial fix. This is not
   optional for contributions that touch architecture, add new types, or change
   public API. The issue is where the design is discussed.

2. **Fork the repository** and create a branch named after the issue:
   `fix/12-slot-variance-validation` or `feature/15-add-vbundle-macro`

3. **Write tests first** if possible. The test suite in `test/runtests.jl` is
   the ground truth for correct behavior.

4. **Follow the code style conventions** above. PRs that ignore style will be
   asked to fix formatting before review.

5. **Update docstrings** for any changed public API. If you add an exported
   symbol, it needs a docstring and a test.

6. **Update `docs/src/index.md`** if you add or remove exported symbols.

7. **Run the full test suite** before opening the PR:

```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
```

8. **Open the PR** against the `main` branch with a clear description of what
   it does and a reference to the issue it addresses.

PRs that break existing tests will not be merged.

---

## Running CI Locally

CI runs three things: tests, documentation build, and doctests. Replicate all
three before pushing:

**Tests:**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Documentation build:**

```bash
julia --project=docs docs/make.jl
```

**Doctests:**

```bash
julia --project=docs --color=yes -e '
using Documenter: DocMeta, doctest
using SymbolicTensors
DocMeta.setdocmeta!(SymbolicTensors, :DocTestSetup, :(using SymbolicTensors); recursive=true)
doctest(SymbolicTensors)
'
```

If all three pass locally, CI will pass.

---

## Code of Conduct

This project follows the
[Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating you agree to abide by its terms. Report unacceptable behavior
to the maintainer directly via GitHub.

---
