# =========================================
# abstractTensor.jl — SymbolicTensors.jl
#
# Defines the AbstractTensor supertype and the KroneckerDelta singleton.
#
# Load order: AbstractTensor is declared in types.jl; this file loads after
#             tensors.jl and metrics.jl, before tensorComponents.jl
#
# AbstractTensor interface — every subtype must implement:
#   print_as(T)   → String   display label
#   tensor_id(T)  → Int      registration order for canonical sort (δ = 0)
#
# Dispatch table for Base.getindex:
#   getindex(::Tensor,        idxs...) → in tensorComponents.jl (Tensor-specific)
#   getindex(::KroneckerDelta, idxs...) → in this file (KroneckerDelta-specific)
# Both return a TensorComponent (defined in tensorComponents.jl).
#
# Shared validation helpers defined here are called by both getindex methods.
# =========================================


# =========================================
# 1.  AbstractTensor interface
# =========================================

"""
    print_as(T::AbstractTensor) -> String

Return the display label of `T`. Used in `show`, error messages, and
[`TensorComponent`](@ref).

Implemented by each concrete subtype.
"""
print_as(T::AbstractTensor) =
    error("$(typeof(T)) must implement print_as(T::AbstractTensor) -> String")

print_as(t::Tensor) = getfield(t, :print_as)

"""
    tensor_id(T::AbstractTensor) -> Int

Internal registration id for canonical ordering of tensor heads.
[`kronecker_delta`](@ref) is always `0`; registered [`Tensor`](@ref)s receive
monotonic ids from [`@def_tensor`](@ref) / [`@def_metric`](@ref).
"""
tensor_id(T::AbstractTensor) =
    error("$(typeof(T)) must implement tensor_id(T::AbstractTensor) -> Int")

tensor_id(t::Tensor) = getfield(t, :tensor_id)


# =========================================
# 2.  KroneckerDelta singleton
# =========================================

"""
    KroneckerDelta

The package-level Kronecker delta (identity tensor) singleton.

`kronecker_delta` is the single global instance of this type. It is defined
at package load time and requires no manifold or vbundle registration.

By convention, components are always written **δ^i_j**: contravariant index
first, covariant index second. Variance is fixed at construction via each
index's vbundle; indices are **not** raised or lowered, and metrics registered
on other [`Tensor`](@ref) heads do not apply to `kronecker_delta`.

At runtime, `[idx_up, idx_down]` must satisfy:
- Both indices are of the same kind (`CoordinateIndex` or `FrameIndex`)
- `idx_up` is contravariant: `_VBUNDLES[idx_up.vbundle].isref == true`
- `idx_down` is covariant: `_VBUNDLES[idx_down.vbundle].isref == false`
- They share the same vbundle of reference:
  `idx_up.vbundle == _VBUNDLES[idx_down.vbundle].dual`

### Usage

    kronecker_delta[a1, -a2]   # identity on tangentM (δ^{a1}_{a2})
    kronecker_delta[B1, -B2]   # identity on E
    kronecker_delta[-a1, a2]   # error: wrong slot order (even if a metric exists)
    kronecker_delta[a1, -B2]   # error: mixed index kinds
    kronecker_delta[a1, -a1]   # valid: same symbol allowed (trace = dim)

### Mathematical meaning

`δ^a_b` is the identity endomorphism `id_V : V → V` for any vector space
`V`. It satisfies `δ^a{}_b T^b = T^a` under contraction (future).
"""
struct KroneckerDelta <: AbstractTensor end

"""
    kronecker_delta

The global Kronecker delta singleton. See [`KroneckerDelta`](@ref).
"""
const kronecker_delta = KroneckerDelta()

print_as(::KroneckerDelta) = "δ"

tensor_id(::KroneckerDelta) = 0


# =========================================
# 3.  Shared index validation helpers
# =========================================
# These are called by both getindex(::Tensor, ...) in tensorComponents.jl
# and getindex(::KroneckerDelta, ...) below.

"""
    _validate_index_registered(idx::AbstractIndex, context::String)

Error if `idx.symbol` is not in the index registries.
`context` is the calling function name for the error message.
"""
function _validate_index_registered(idx::AbstractIndex, context::String)
    is_index_registered(idx.symbol) ||
        error("$context: index :$(idx.symbol) is not registered.")
end

"""
    _validate_index_vbundle_registered(idx::AbstractIndex, context::String)

Error if `idx.vbundle` is not in `_VBUNDLES`.
"""
function _validate_index_vbundle_registered(idx::AbstractIndex, context::String)
    haskey(_VBUNDLES, idx.vbundle) ||
        error(
            "$context: index :$(idx.symbol) has unregistered " *
            "vbundle :$(idx.vbundle)."
        )
end

"""
    _validate_index_on_manifold(idx::AbstractIndex, manifold_sym::Symbol, context::String)

Error if `idx.vbundle` does not belong to `manifold_sym`.
"""
function _validate_index_on_manifold(
    idx          :: AbstractIndex,
    manifold_sym :: Symbol,
    context      :: String,
)
    _VBUNDLES[idx.vbundle].manifold == manifold_sym ||
        error(
            "$context: index :$(idx.symbol) is on manifold " *
            ":$(_VBUNDLES[idx.vbundle].manifold), but expected manifold " *
            ":$(manifold_sym)."
        )
end

"""
    _validate_index_vbundle_of_reference(idx::AbstractIndex, ref_vb::Symbol, tensor_label::String, context::String)

Error if `idx` does not derive from `ref_vb` (the tensor's vbundle of reference).
"""
function _validate_index_vbundle_of_reference(
    idx          :: AbstractIndex,
    ref_vb       :: Symbol,
    tensor_label :: String,
    context      :: String,
)
    _vbundle_of_reference_of(idx.vbundle) == ref_vb ||
        error(
            "$context: index :$(idx.symbol) has vbundle of reference " *
            ":$(_vbundle_of_reference_of(idx.vbundle)), but tensor " *
            "$(tensor_label) has vbundle of reference :$ref_vb."
        )
end


# =========================================
# 4.  KroneckerDelta getindex
# =========================================

"""
    Base.getindex(δ::KroneckerDelta, idxs...) -> TensorComponent

Construct a [`TensorComponent`](@ref) for the Kronecker delta.

**Convention:** always `kronecker_delta[idx_up, idx_down]` for δ^i_j. Slot
variance is not relaxed by metrics on other tensors; use unary `-` (`flip`) on
indices or reorder slots instead of expecting metric raising/lowering here.

Validates:
1. Exactly two indices.
2. Both indices are the same kind (`CoordinateIndex` or `FrameIndex`).
3. First index is contravariant (`isref == true`).
4. Second index is covariant (`isref == false`).
5. They share the same vbundle of reference.
6. Both are registered and their vbundles are registered.

# Examples
~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
@def_metric g tangentM
kronecker_delta[a1, -a2]    # identity on tangentM
g[a1, a2]                   # valid for g (metric relaxes slot variance)
kronecker_delta[-a1, a2]    # error: wrong δ slot order (metric does not apply)

@def_vbundle E M 3 [B1, B2, B3]
kronecker_delta[B1, -B2]    # identity on E

kronecker_delta[a1, -B2]    # error: mixed index kinds (coord vs frame)
kronecker_delta[a1, -a2, a3] # error: requires exactly 2 indices
~~~
"""
function Base.getindex(δ::KroneckerDelta, idxs...)
    n = length(idxs)
    n == 2 ||
        error(
            "KroneckerDelta: requires exactly 2 indices, got $n."
        )

    ctx = "KroneckerDelta"

    # Parse arguments to AbstractIndex
    idx1 = _parse_index_arg(idxs[1])
    idx2 = _parse_index_arg(idxs[2])

    # Validate registration
    _validate_index_registered(idx1, ctx)
    _validate_index_registered(idx2, ctx)
    _validate_index_vbundle_registered(idx1, ctx)
    _validate_index_vbundle_registered(idx2, ctx)

    # Same index kind
    typeof(idx1) === typeof(idx2) ||
        error(
            "KroneckerDelta: both indices must be the same kind " *
            "(both CoordinateIndex or both FrameIndex). " *
            "Got $(typeof(idx1)) and $(typeof(idx2))."
        )

    # First index contravariant (isref == true)
    _VBUNDLES[idx1.vbundle].isref ||
        error(
            "KroneckerDelta: by convention δ is written δ^i_j — contravariant " *
            "index first, covariant second. The first index :$(idx1.symbol) must be " *
            "contravariant (reference vbundle, isref=true); got :$(idx1.vbundle) " *
            "(isref=false). Indices cannot be raised or lowered by a metric. " *
            "Did you mean kronecker_delta[$(idx2.symbol), -$(idx1.symbol)] for the " *
            "canonical slot order?"
        )

    # Second index covariant (isref == false)
    !_VBUNDLES[idx2.vbundle].isref ||
        error(
            "KroneckerDelta: by convention δ is written δ^i_j — contravariant " *
            "index first, covariant second. The second index :$(idx2.symbol) must be " *
            "covariant (dual vbundle, isref=false); got :$(idx2.vbundle) " *
            "(isref=true). Use -$(idx2.symbol) for the covariant form. " *
            "Metrics on other tensors do not relax this rule."
        )

    # Same vbundle of reference
    ref1 = idx1.vbundle                           # already isref=true, so this is the ref vb
    ref2 = _vbundle_of_reference_of(idx2.vbundle) # dual → look up its ref
    ref1 == ref2 ||
        error(
            "KroneckerDelta: indices :$(idx1.symbol) and :$(idx2.symbol) " *
            "derive from different vbundles of reference: :$ref1 and :$ref2. " *
            "The Kronecker delta requires both indices to share the same " *
            "vbundle of reference."
        )

    # Both indices on the same manifold
    m1 = _VBUNDLES[idx1.vbundle].manifold
    m2 = _VBUNDLES[idx2.vbundle].manifold
    m1 == m2 ||
        error(
            "KroneckerDelta: index :$(idx1.symbol) is on manifold :$m1 " *
            "but index :$(idx2.symbol) is on manifold :$m2."
        )

    TensorComponent(δ, [idx1, idx2])
end


# =========================================
# 5.  AbstractTensor predicates
# =========================================

"""
    is_abstract_tensor(x) -> Bool

Return `true` if `x` is an [`AbstractTensor`](@ref) instance
(i.e. a [`Tensor`](@ref) or a [`KroneckerDelta`](@ref)).
"""
is_abstract_tensor(x) = x isa AbstractTensor

"""
    is_kronecker_delta(x) -> Bool

Return `true` if `x` is the [`KroneckerDelta`](@ref) singleton.
"""
is_kronecker_delta(x) = x isa KroneckerDelta


# =========================================
# 6.  show methods for KroneckerDelta
# =========================================

function Base.show(io::IO, ::MIME"text/plain", ::KroneckerDelta)
    print(io, "KroneckerDelta δ  (global singleton, vbundle-agnostic)")
end

function Base.show(io::IO, ::MIME"text/html", ::KroneckerDelta)
    print(io,
        "<span style=\"font-style:italic;\">δ</span> " *
        "<span style=\"color:#888;font-size:0.9em;\">" *
        "(Kronecker delta, vbundle-agnostic)</span>"
    )
end

function Base.show(io::IO, ::MIME"text/latex", ::KroneckerDelta)
    print(io, "\$\\delta\$")
end


# =========================================
# Exports
# =========================================

export AbstractTensor
export KroneckerDelta, kronecker_delta
export print_as
export is_abstract_tensor, is_kronecker_delta
export _validate_index_registered
export _validate_index_vbundle_registered
export _validate_index_on_manifold
export _validate_index_vbundle_of_reference