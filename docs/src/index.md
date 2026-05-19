```@meta
CurrentModule = AbstractTensors
```

# AbstractTensors.jl

Documentation for [AbstractTensors.jl](https://github.com/THelpin/AbstractTensors.jl).

AbstractTensors.jl is a Julia package for symbolic tensor algebra on
differentiable manifolds. It provides the infrastructure needed to define
tensors with their slot structure and symmetry groups, perform Einstein
summation, raise/lower indices, and work with general (possibly
non-metric-compatible) connections and symplectic structures.

## Design overview

**Registries.** Every manifold, vector bundle, index
symbol, metric, and tensor is stored in module-level dictionaries (`_MANIFOLDS`,
`_VBUNDLES`, `_INDICES`, `_TENSORS`, `_METRICS`). These are populated by
the `@def_*` macros and cleaned up by the corresponding `@undef_*` macros.

**Instance-based objects.** [`Manifold`](@ref), [`VBundle`](@ref), and
[`Tensor`](@ref) are plain structs. Each `@def_*` macro binds the new object
as a variable in the caller's scope and registers it. All metadata is
queryable via dot access:

    M.dim              # 4
    M.tangent_bundle   # :tangentM
    M.cotangent_bundle # :cotangentM
    T.rank             # 2
    T.symmetry         # SlotSymmetry(n=2, order=2, ngens=1)

**Tensor indices.** A [`TensorIndex`](@ref) is a `(symbol, vbundle)` pair.
Whether an index is contravariant or covariant is determined entirely by
which bundle it lives in. [`VBundle.isdual`](@ref VBundle) is the single
authoritative source; no naming convention is relied upon.

**Index symbols.** Each index is bound to a contravariant [`TensorIndex`](@ref)
in the caller's scope by `@def_manifold` and `@add_indices`, enabling dot
access and bracket sugar via unary `-` / `+`:

    a1.symbol    # :a1
    a1.vbundle   # :tangentM
    a1           # TensorIndex(:a1, :tangentM)   — contravariant
    -a1          # TensorIndex(:a1, :cotangentM) — covariant (unary -)
    flip(a1)     # toggle variance

**Slot structure.** Each tensor stores its slot structure as a
`Vector{Symbol}` of vbundle names — one per slot. The sign prefixes in
`@def_tensor T[-a1, a2] M` communicate variance at definition time and are
then discarded; the tensor retains only `[:cotangentM, :tangentM]`. Index
symbols used at definition time are validated against the manifold and
forgotten.

**Symmetry groups.** Tensor slot symmetries are represented as
[`SlotSymmetry`](@ref) objects: subgroups of the signed permutation group
\(\mathbb{Z}\_2 \wr S_n\). Each group element is a [`SignedPerm`](@ref) —
a permutation of slot positions paired with a sign `±1` that records the
scalar factor the tensor acquires under that reordering. The group is
stored as its complete closed element set, computed from generators at
construction time via BFS — exact and fast for all physical tensor ranks.

## Workflow

```julia
using AbstractTensors

# 1. Define a 4-dimensional manifold with index symbols
@def_manifold M 4 [a1, a2, a3, a4]

# 2. Query manifold and bundle metadata
M.dim              # 4
M.tangent_bundle   # :tangentM
tangentM.isdual    # false
a1.vbundle         # :tangentM

# 3. Contravariant and covariant indices (bound by @def_manifold)
a1        # TensorIndex(:a1, :tangentM)    — contravariant
-a1       # TensorIndex(:a1, :cotangentM)  — covariant
F[-a1, -a2]   # bracket indexing uses TensorIndex values only

# 4. Define tensors with varying slot structures and symmetries
@def_tensor T[-a1, -a2] M                          # rank-2 covariant, no symmetry
@def_tensor g[-a1, -a2] M symmetry=symmetric(2)   # symmetric metric
@def_tensor A[-a1, -a2] M symmetry=antisymmetric(2)  # 2-form

@def_tensor ε[-a1,-a2,-a3,-a4] M symmetry=antisymmetric(4)        # Levi-Civita
@def_tensor R[-a1,-a2,-a3,-a4] M symmetry=riemann_symmetry()      # Riemann
@def_tensor W[-a1,-a2,-a3,-a4] M symmetry=riemann_symmetry() traceless=true print_as=:Weyl

# Mixed (1,1) tensor
@def_tensor Γ[a1, -a2] M

# 5. Inspect a tensor
T.rank         # 2
T.manifold     # :M
T.slots        # [:cotangentM, :cotangentM]
T.symmetries     # NoSymmetry(n=2)
T.is_traceless # false
T.print_as     # :T
tensor_info(R)

# 6. Canonical form of an index list under a symmetry
sym  = antisymmetric(2)
a, b = TensorIndex(:a1, :tangentM), TensorIndex(:a2, :tangentM)
canonical_rep([b, a], sym)   # ([a, b], Int8(-1)) — T[b,a] = -T[a,b]

# 7. Add extra indices after definition
@add_indices M a5 a6

# 8. Remove objects from the registries
@undef_tensor T
@undef_manifold M
```

---

## Indices

```@docs
TensorIndex
flip(::TensorIndex)
is_up
is_down
@add_indices
```

---

## Manifolds

```@docs
Manifold
@def_manifold
@undef_manifold
```

---

## Vector bundles

```@docs
VBundle
@def_vbundle
@undef_vbundle
```

---

## Signed permutations

```@docs
SignedPerm
identity_perm
is_valid_perm
compose
apply
```

---

## Slot symmetry groups

```@docs
SlotSymmetry
no_symmetry
symmetric
antisymmetric
symmetric_on
antisymmetric_on
riemann_symmetry
is_in_symmetry
is_trivial_symmetry
canonical_rep
```

---

## Tensors

```@docs
Tensor
@def_tensor
@undef_tensor
is_tensor
list_tensors
tensor_info
```

---

## Registries

```@docs
_MANIFOLDS
_VBUNDLES
_INDICES
_TENSORS
_METRICS
show_registry
```

---

## Internal / developer API

These functions are used internally by the package and are not part of the
public API. They are documented here for contributors.

```@docs
contractable
validate_contraction
validate_indices
register_index!
unregister_index!
index_home_vbundle
is_dual_vbundles
list_manifolds
```

---

## Full API index

```@index

```
