```@meta
CurrentModule = AbstractTensors
```

# AbstractTensors.jl

Documentation for [AbstractTensors.jl](https://github.com/THelpin/AbstractTensors.jl).

AbstractTensors.jl is a Julia package for symbolic tensor algebra on
differentiable manifolds. It provides a mathematically rigorous index
system, a manifold registry, and the infrastructure needed to define
tensors, perform Einstein summation, raise/lower indices, and work with
general (possibly non-metric-compatible) connections and symplectic structures.

## Design overview

The package is built around two core ideas:

**Registries over global state.** Every manifold, vector bundle, and index
symbol is stored in module-level dictionaries (`_MANIFOLDS`, `_VBUNDLES`,
`_IDX_REGISTRY`). There are no global variables that silently change —
every mutation goes through an explicit API call.

**Bundle membership encodes variance.** A [`TensorIndex`](@ref) is a
`(symbol, vbundle)` pair. Whether an index is contravariant or covariant
is determined entirely by which bundle it lives in — `:TangentM` for upper,
`:CoTangentM` for lower. The field [`VBundleRecord.isdual`](@ref VBundleRecord)
is the single authoritative source; no naming convention is assumed.

## Workflow

```julia
using AbstractTensors

# 1. Define a 4-dimensional manifold with four index symbols
@def_manifold M 4 [a1, a2, a3, a4]

# 2. Construct contravariant and covariant indices
up(:a1)    # TensorIndex(:a1, :TangentM)   — contravariant
down(:a1)  # TensorIndex(:a1, :CoTangentM) — covariant

# 3. Flip an index between bundles (e.g. when applying a metric)
flip(up(:a1))   # TensorIndex(:a1, :CoTangentM)

# 4. Test contractability
contractable(up(:a1), down(:a1))   # true
contractable(up(:a1), up(:a1))     # false — same bundle

# 5. Add extra indices after definition
@add_indices M a5 a6

# 6. Remove a manifold and all its registrations
@undef_manifold M
```

---

## Index types

```@docs
TensorIndex
up
down
flip(::TensorIndex)
symbol_of
vbundle_of
is_contravariant
is_covariant
is_up
is_down
same_symbol
contractable
dual_vbundles
validate_contraction
validate_indices
```

---

## Index registry

```@docs
_IDX_REGISTRY
register_index!
unregister_index!
index_registered
index_home_vbundle
```

---

## Manifolds

```@docs
Manifold
ManifoldRecord
@def_manifold
@undef_manifold
is_manifold
dim_of_manifold
tangent_bundle_of
cotangent_bundle_of
vbundles_of
base_manifold
list_manifolds
manifold_info
```

---

## Vector bundles

```@docs
VBundle
TangentBundle
CoTangentBundle
VBundleRecord
dual_bundle
is_vbundle
is_tangent_bundle
is_cotangent_bundle
indices_of_vbundle
```

---

## Registries

```@docs
_MANIFOLDS
_VBUNDLES
show_registry
```

---

## Extra indices

```@docs
@add_indices
```

---

## Full API index

```@index

```
