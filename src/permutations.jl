# =========================================
# permutations.jl — AbstractTensors.jl
#
# — SignedPerm
#   A permutation of 1:n paired with a sign ±1.
#   Convention (pull-back / preimage): images[i] = j means
#   "position i of the result draws from position j of the source",
#   so  apply(g, v)[i] = v[g.images[i]].
#   Composition:  compose(f, g) = f ∘ g  (apply g first, then f):
#     compose(f,g).images[i] = g.images[f.images[i]]
#     compose(f,g).sign      = f.sign * g.sign
#
# — SlotSymmetry
#   The symmetry group of a tensor's slot positions.
#   Stored as the complete closed set of group elements, computed from
#   generators via BFS at construction time.
#   Exact and fast for all physical tensor cases (rank ≤ 8, |G| ≤ ~10^4).
#   No external dependencies.
#
# xTensor analogs:
#   SignedPerm                  ↔  a signed permutation in the Schreier–Sims tree
#   SlotSymmetry                ↔  Symmetric[T] / StrongGenSet[...]
#   symmetric(n)                ↔  Symmetric[{i1,...,in}]
#   antisymmetric(n)            ↔  Antisymmetric[{i1,...,in}]
#   canonical_rep(indices, sym) ↔  CanonicalPerm / index sorting by symmetry
# =========================================


# =========================================
# 1.  SignedPerm
# =========================================

"""
    SignedPerm

A permutation of positions `1:n` paired with a sign `±1`.

Pull-back convention: `images[i] = j` means "position `i` of the result
draws from position `j` of the source". Consequently:

    apply(g, v)[i] = v[g.images[i]]

and composition is defined as:

    compose(f, g) = f ∘ g   (g applied first, then f)
    compose(f,g).images[i] = g.images[f.images[i]]
    compose(f,g).sign       = f.sign * g.sign

The sign `±1` records the scalar factor the tensor picks up under this
slot reordering (e.g. `+1` for symmetric, `-1` for a single transposition
in an antisymmetric tensor).

Construction
------------
    SignedPerm([2, 1], Int8(1))    # swap positions 1↔2, no sign change
    SignedPerm([2, 1], Int8(-1))   # swap positions 1↔2, flip sign
    identity_perm(n)               # identity permutation of degree n

Fields
------
- `images` : pull-back image vector, a permutation of `1:n`
- `sign`   : scalar factor, must be `+1` or `-1`
"""
struct SignedPerm
    images::Vector{Int}
    sign::Int8
end

# ── Equality and hashing ─────────────────────────────────────────────────────

Base.:(==)(a::SignedPerm, b::SignedPerm) =
    a.images == b.images && a.sign == b.sign

Base.hash(g::SignedPerm, h::UInt) = hash((g.images, g.sign), h)

# ── Validation ───────────────────────────────────────────────────────────────

"""
    is_valid_perm(g::SignedPerm) -> Bool

Return `true` if `g` is a valid signed permutation:
- `images` is a permutation of `1:n` (no repeats, all values in range)
- `sign` is `+1` or `-1`
"""
function is_valid_perm(g::SignedPerm)
    n = length(g.images)
    g.sign in (Int8(1), Int8(-1)) || return false
    sort(g.images) == collect(1:n)
end

# ── Constructors ─────────────────────────────────────────────────────────────

"""
    identity_perm(n::Int) -> SignedPerm

Return the identity signed permutation of degree `n` (sign `+1`).
"""
identity_perm(n::Int) = SignedPerm(collect(1:n), Int8(1))

# ── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, g::SignedPerm)
    sign_str = g.sign == Int8(1) ? "+" : "-"
    print(io, "SignedPerm($(g.images), $(sign_str)1)")
end


# =========================================
# 2.  Composition, inverse, application
# =========================================

"""
    compose(f::SignedPerm, g::SignedPerm) -> SignedPerm

Return `f ∘ g`: apply `g` first, then `f`.

    compose(f, g).images[i] = g.images[f.images[i]]
    compose(f, g).sign      = f.sign * g.sign

Equivalent to `apply(f, apply(g, v))` for any vector `v`.
"""
function compose(f::SignedPerm, g::SignedPerm)
    length(f.images) == length(g.images) ||
        error("compose: permutations have different degrees " *
              "$(length(f.images)) ≠ $(length(g.images))")
    n = length(f.images)
    SignedPerm([g.images[f.images[i]] for i in 1:n], Int8(f.sign * g.sign))
end

"""
    Base.inv(g::SignedPerm) -> SignedPerm

Return the group-theoretic inverse of `g`.
The inverse permutation satisfies `compose(g, inv(g)) = identity_perm(n)`.
The sign of the inverse equals the sign of `g` (since `g.sign^2 = 1`).
"""
function Base.inv(g::SignedPerm)
    n = length(g.images)
    inv_imgs = Vector{Int}(undef, n)
    for i in 1:n
        inv_imgs[g.images[i]] = i
    end
    SignedPerm(inv_imgs, g.sign)
end

"""
    apply(g::SignedPerm, v::Vector) -> Vector

Reorder `v` according to `g`: `result[i] = v[g.images[i]]`.

Does not apply the sign (the sign is a scalar factor for the tensor
expression, not a transformation of the index list itself).
"""
function apply(g::SignedPerm, v::Vector{T}) where {T}
    length(v) == length(g.images) ||
        error("apply: vector length $(length(v)) ≠ permutation degree $(length(g.images))")
    [v[g.images[i]] for i in 1:length(g.images)]
end


# =========================================
# 3.  Internal slot-swap helper
# =========================================

# Builds a SignedPerm of degree n that swaps positions i and j,
# identity elsewhere. Called by symmetric_on / antisymmetric_on.
function _swap_perm(n::Int, i::Int, j::Int, sign::Int8)
    (1 <= i <= n && 1 <= j <= n && i != j) ||
        error("_swap_perm: positions $i and $j must be distinct values in 1:$n")
    imgs    = collect(1:n)
    imgs[i] = j
    imgs[j] = i
    SignedPerm(imgs, sign)
end


# =========================================
# 4.  BFS group closure
# =========================================

"""
    close_group(generators::Vector{SignedPerm}, n::Int) -> Vector{SignedPerm}

!!! warning "Internal"
    Called by the `SlotSymmetry` constructor. Not part of the public API.

Compute the full group generated by `generators` via BFS: start from the
identity, then repeatedly right-multiply by each generator and its inverse
until no new elements are found.

The result is exact and terminates for any finite group. For typical tensor
symmetry groups (|G| ≤ a few thousand) this is fast and allocation-cheap.
"""
function close_group(generators::Vector{SignedPerm}, n::Int)
    id   = identity_perm(n)
    seen = Set{SignedPerm}([id])
    queue = SignedPerm[id]

    # Include both generators and their inverses to ensure full closure
    # even when generators are not self-inverse.
    all_gens = SignedPerm[]
    for g in generators
        push!(all_gens, g)
        gi = inv(g)
        gi ∉ all_gens && push!(all_gens, gi)
    end

    while !isempty(queue)
        g = popfirst!(queue)
        for gen in all_gens
            h = compose(g, gen)
            if h ∉ seen
                push!(seen, h)
                push!(queue, h)
            end
        end
    end

    return collect(seen)
end


# =========================================
# 5.  SlotSymmetry
# =========================================

"""
    SlotSymmetry

The symmetry group of a tensor's slot positions, represented as:

- `n`              : number of slots (degree of the permutation group)
- `generators`     : a generating set of `SignedPerm`s
- `group_elements` : the complete closed set of group elements, computed
                     from `generators` at construction time via BFS

The group acts on slot *positions* `1:n`, not on index symbols.
The sign of each element records the scalar factor the tensor acquires
under that slot reordering (e.g. `−1` for a transposition in an
antisymmetric tensor).

Construction
------------
Use the outer constructor `SlotSymmetry(n, generators)` which validates
inputs and computes the closure automatically.  Convenience constructors
are also provided:

    no_symmetry(n)                    # trivial group {id}
    symmetric(n)                      # fully symmetric on all n slots
    antisymmetric(n)                  # fully antisymmetric on all n slots
    symmetric_on(n, positions)        # symmetric on specified slot subset
    antisymmetric_on(n, positions)    # antisymmetric on specified slot subset
    riemann_symmetry()                # standard algebraic symmetry of R_{abcd}

Fields
------
- `n`              : slot count / permutation degree
- `generators`     : the user-supplied generating set
- `group_elements` : all group elements (closed under composition)
"""
struct SlotSymmetry
    n::Int
    generators::Vector{SignedPerm}
    group_elements::Vector{SignedPerm}
end

"""
    SlotSymmetry(n::Int, generators::Vector{SignedPerm}) -> SlotSymmetry

Outer constructor: validate generators and compute the full group closure.
"""
function SlotSymmetry(n::Int, generators::Vector{SignedPerm})
    n >= 1 || error("SlotSymmetry: n must be ≥ 1, got $n")
    for (k, g) in enumerate(generators)
        length(g.images) == n ||
            error("SlotSymmetry: generator $k has degree $(length(g.images)), expected $n")
        is_valid_perm(g) ||
            error("SlotSymmetry: generator $k is not a valid signed permutation: $g")
    end
    elements = close_group(generators, n)
    SlotSymmetry(n, generators, elements)
end

# ── Predicates ───────────────────────────────────────────────────────────────

"""
    is_in_symmetry(g::SignedPerm, sym::SlotSymmetry) -> Bool

Return `true` if `g` is an element of the symmetry group `sym`.
"""
is_in_symmetry(g::SignedPerm, sym::SlotSymmetry) = g ∈ sym.group_elements

"""
    is_trivial_symmetry(sym::SlotSymmetry) -> Bool

Return `true` if `sym` is the trivial group (only the identity element).
"""
is_trivial_symmetry(sym::SlotSymmetry) = length(sym.group_elements) == 1

# ── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, sym::SlotSymmetry)
    ord = length(sym.group_elements)
    if ord == 1
        print(io, "NoSymmetry(n=$(sym.n))")
    else
        print(io, "SlotSymmetry(n=$(sym.n), order=$ord, ngens=$(length(sym.generators)))")
    end
end


# =========================================
# 6.  Convenience constructors
# =========================================

"""
    no_symmetry(n::Int) -> SlotSymmetry

Trivial symmetry group: only the identity element.
Use for tensors with no slot permutation symmetry.
"""
no_symmetry(n::Int) = SlotSymmetry(n, SignedPerm[])

"""
    symmetric(n::Int) -> SlotSymmetry

Fully symmetric group on all `n` slots: every permutation has sign `+1`.
The group order is `n!`.  Generated by adjacent transpositions `(i, i+1)`
with sign `+1`.
"""
function symmetric(n::Int)
    n >= 1 || error("symmetric: n must be ≥ 1")
    gens = [_swap_perm(n, i, i + 1, Int8(1)) for i in 1:n-1]
    SlotSymmetry(n, gens)
end

"""
    antisymmetric(n::Int) -> SlotSymmetry

Fully antisymmetric group on all `n` slots: every permutation has sign
equal to its parity `(−1)^{inv(σ)}`.  The group order is `n!`.
Generated by adjacent transpositions `(i, i+1)` with sign `−1`.
"""
function antisymmetric(n::Int)
    n >= 1 || error("antisymmetric: n must be ≥ 1")
    gens = [_swap_perm(n, i, i + 1, Int8(-1)) for i in 1:n-1]
    SlotSymmetry(n, gens)
end

"""
    symmetric_on(n::Int, positions::Vector{Int}) -> SlotSymmetry

Symmetry group that acts symmetrically (sign `+1`) on the slot positions
listed in `positions`, leaving all other slots fixed.

    symmetric_on(4, [1, 2])   # T_{abcd} symmetric in first two slots
"""
function symmetric_on(n::Int, positions::Vector{Int})
    isempty(positions) && return no_symmetry(n)
    all(1 .<= positions .<= n) ||
        error("symmetric_on: positions $positions out of range 1:$n")
    gens = [_swap_perm(n, positions[i], positions[i+1], Int8(1))
            for i in 1:length(positions)-1]
    SlotSymmetry(n, gens)
end

"""
    antisymmetric_on(n::Int, positions::Vector{Int}) -> SlotSymmetry

Symmetry group that acts antisymmetrically (sign `−1` per transposition)
on the slot positions listed in `positions`, leaving all other slots fixed.

    antisymmetric_on(4, [1, 2])   # T_{abcd} antisymmetric in first two slots
"""
function antisymmetric_on(n::Int, positions::Vector{Int})
    isempty(positions) && return no_symmetry(n)
    all(1 .<= positions .<= n) ||
        error("antisymmetric_on: positions $positions out of range 1:$n")
    gens = [_swap_perm(n, positions[i], positions[i+1], Int8(-1))
            for i in 1:length(positions)-1]
    SlotSymmetry(n, gens)
end

"""
    riemann_symmetry() -> SlotSymmetry

The standard algebraic slot symmetry of the Riemann tensor `R_{abcd}`:

- Antisymmetric in slots (1, 2): `R_{abcd} = −R_{bacd}`
- Antisymmetric in slots (3, 4): `R_{abcd} = −R_{abdc}`
- Symmetric under pair exchange:  `R_{abcd} =  R_{cdab}`

Group order: 8.  Use as `symmetry=riemann_symmetry()` in `@def_tensor`.
"""
function riemann_symmetry()
    gens = SignedPerm[
        SignedPerm([2, 1, 3, 4], Int8(-1)),   # antisym in slots (1,2)
        SignedPerm([1, 2, 4, 3], Int8(-1)),   # antisym in slots (3,4)
        SignedPerm([3, 4, 1, 2], Int8(1)),    # sym under pair swap (1,2)↔(3,4)
    ]
    SlotSymmetry(4, gens)
end


# =========================================
# 7.  AbstractIndex ordering  (used by canonical_rep)
# =========================================

# Total order on AbstractIndex for lexicographic comparison of index lists.
# CoordinateIndex sorts before BasisIndex; then compare (symbol, vbundle).
function _index_sort_key(idx::AbstractIndex)
    kind = idx isa CoordinateIndex ? 0 : 1
    (kind, string(idx.symbol), string(idx.vbundle))
end

Base.isless(a::AbstractIndex, b::AbstractIndex) =
    _index_sort_key(a) < _index_sort_key(b)


# =========================================
# 8.  Canonical representative
# =========================================

"""
    canonical_rep(indices::Vector{T}, sym::SlotSymmetry)
        -> (canonical::Vector{T}, sign::Int8)

Find the canonical (lexicographically smallest) representative of
`indices` under the action of `sym`, and return it together with the
sign of the group element that produced it.

The sign tells you the scalar factor: if the canonical permutation `g`
satisfies `apply(g, indices) = canonical`, then

    T[indices...] = g.sign * T[canonical...]

Any element type `T` with `isless` defined is supported.  For
[`AbstractIndex`](@ref), `isless` is defined in this file.

# Example
```julia
sym  = antisymmetric(2)
a, b = CoordinateIndex(:a, :TangentM), CoordinateIndex(:b, :TangentM)
canonical_rep([b, a], sym)   # ([a, b], Int8(-1))
# i.e. T[b, a] = -T[a, b]
```
"""
function canonical_rep(indices::Vector{T}, sym::SlotSymmetry) where {T}
    length(indices) == sym.n ||
        error("canonical_rep: $(length(indices)) indices but symmetry degree is $(sym.n)")

    best_indices = indices
    best_sign    = Int8(1)

    for g in sym.group_elements
        candidate = apply(g, indices)
        if candidate < best_indices
            best_indices = candidate
            best_sign    = g.sign
        end
    end

    return best_indices, best_sign
end


# =========================================
# Exports
# =========================================

export SignedPerm
export is_valid_perm, identity_perm
export compose, apply
export SlotSymmetry
export is_in_symmetry, is_trivial_symmetry
export no_symmetry, symmetric, antisymmetric
export symmetric_on, antisymmetric_on
export riemann_symmetry
export canonical_rep
