# =========================================
# tensorExpressions.jl — AbstractTensors.jl
#
# A TensorExpression is a Tensor (schema) applied to a specific list of
# TensorIndex objects. It is the REPL/notebook object you interact with:
#
#   F[-a1, -a2]             — sugar: -a1 / +a1 on TensorIndex (indices.jl)
#
# The struct is lazy and inert: no contraction, no symmetry reduction
# happens at construction time. Algebra (*,+) and contraction are
# implemented in future files (tensorAlgebra.jl, contractions.jl).
#
# Depends on: indices.jl, manifolds.jl, permutations.jl, tensors.jl, metrics.jl
# =========================================


# =========================================
# 1.  TensorExpression struct
# =========================================

"""
    TensorExpression

A [`Tensor`](@ref) (definition-level schema) applied to a concrete list of
[`TensorIndex`](@ref) objects, representing one occurrence in an algebraic
expression.

Constructed via `getindex` on a [`Tensor`](@ref):

    F[-a1, -a2]             # covariant slots via unary - on bound indices
    F[a1, -a2]              # mixed: bare a1 or +a1; -a2 for covariant slot

The expression is **lazy and inert**: no contraction, canonicalization, or
symmetry reduction is performed at construction time.

Fields
------
- `tensor`  : the [`Tensor`](@ref) this expression refers to
- `indices` : the concrete index list for this occurrence, one per slot
"""
struct TensorExpression
    tensor::Tensor
    indices::Vector{TensorIndex}
end


# =========================================
# 2.  Internal argument parser
# =========================================

# Accepts TensorIndex only (includes -a1 / +a1 after indices.jl unary ops).
function _parse_index_arg(arg)::TensorIndex
    arg isa TensorIndex && return arg
    error(
        "TensorExpression: slot argument must be a TensorIndex " *
        "(e.g. a1 or -a1 from @def_manifold), got $(repr(arg)) " *
        "of type $(typeof(arg))."
    )
end


# =========================================
# 3.  Base.getindex — F[-a1, -a2] → TensorExpression
# =========================================

"""
    Base.getindex(T::Tensor, idxs...) -> TensorExpression

Construct a [`TensorExpression`](@ref) by applying `T` to the given indices.

Each slot must be a [`TensorIndex`](@ref) (typically a variable bound by
`@def_manifold` / `@add_indices`, optionally with unary `-` or `+`).

Validates at construction time:
1. **Arity** — `length(idxs) == T.rank`
2. **Manifold membership** — each index's home tangent bundle must match `T.manifold`
3. **Slot variance** — each index's vbundle must match the corresponding slot in `T.slots`

# Examples
```julia
@def_manifold M 4 [a1, a2, a3, a4]
@def_metric g[-a1, -a2] M
@def_tensor F[-a1, -a2] M symmetries=[antisymmetric(2)]

F[-a1, -a2]              # covariant slots
F[a1, -a2]               # mixed (only if F has a contravariant first slot)
F[-a1, -a2, -a1]         # error: rank mismatch
```
"""
function Base.getindex(T::Tensor, idxs...)
    n = length(idxs)

    n == T.rank ||
        error(
            "TensorExpression: tensor $(T.print_as) has rank $(T.rank) " *
            "but $n index argument(s) were given."
        )

    # Step 1: parse each argument to a TensorIndex.
    ti = Vector{TensorIndex}(undef, n)
    for i in 1:n
        ti[i] = _parse_index_arg(idxs[i])
    end

    # Step 2: retrieve manifold metadata for validation.
    haskey(_MANIFOLDS, T.manifold) ||
        error(
            "TensorExpression: tensor $(T.print_as) references " *
            "unregistered manifold :$(T.manifold)."
        )
    tb = _MANIFOLDS[T.manifold].tangent_bundle   # e.g. :tangentM

    # Step 3: per-slot validation.
    for i in 1:n
        idx = ti[i]

        # Manifold membership: index_home_vbundle always returns the tangent
        # bundle, so comparing it to tb is the correct manifold check.
        is_index_registered(idx.symbol) ||
            error(
                "TensorExpression: index :$(idx.symbol) is not registered."
            )
        home = index_home_vbundle(idx.symbol)
        home == tb ||
            error(
                "TensorExpression: index :$(idx.symbol) has home bundle " *
                ":$home, expected :$tb (manifold $(T.manifold))."
            )

        # Variance: the index's actual vbundle must equal the slot's vbundle.
        slot_vb = T.slots[i]
        if idx.vbundle != slot_vb
            cotb = _MANIFOLDS[T.manifold].cotangent_bundle
            sym  = idx.symbol
            hint = slot_vb == cotb ? "-$sym" : "$sym"
            error(
                "TensorExpression: slot $i of $(T.print_as) expects " *
                "bundle :$slot_vb but index :$sym lives in " *
                ":$(idx.vbundle). Use $hint."
            )
        end
    end

    TensorExpression(T, ti)
end


# =========================================
# 4.  Accessors
# =========================================

"""Return the [`Tensor`](@ref) schema of a `TensorExpression`."""
tensor_of(e::TensorExpression)  = e.tensor

"""Return the concrete index list of a `TensorExpression`."""
indices_of_tensor(e::TensorExpression) = e.indices

"""
    rank_of(e::TensorExpression) -> Int

Number of slots of the expression. Dispatches alongside `rank_of(::Tensor)`.
"""
rank_of(e::TensorExpression) = length(e.indices)


# =========================================
# 5.  Equality and hashing
# =========================================

Base.:(==)(a::TensorExpression, b::TensorExpression) =
    a.tensor === b.tensor && a.indices == b.indices

Base.hash(e::TensorExpression, h::UInt) =
    hash((objectid(e.tensor), e.indices), h)


# =========================================
# 6.  Display helpers
# =========================================

# Unicode sub/superscript maps for Latin letters only.
# Digits (0-9) are intentionally omitted and will display at normal height.
# Characters not in the table are kept as-is (normal height fallback).
const _CHAR_TO_SUB = Dict{Char,Char}(
    'a'=>'ₐ','e'=>'ₑ','h'=>'ₕ','i'=>'ᵢ','j'=>'ⱼ',
    'k'=>'ₖ','l'=>'ₗ','m'=>'ₘ','n'=>'ₙ','o'=>'ₒ',
    'p'=>'ₚ','r'=>'ᵣ','s'=>'ₛ','t'=>'ₜ','u'=>'ᵤ',
    'v'=>'ᵥ','x'=>'ₓ',
)

const _CHAR_TO_SUP = Dict{Char,Char}(
    'a'=>'ᵃ','b'=>'ᵇ','c'=>'ᶜ','d'=>'ᵈ','e'=>'ᵉ',
    'f'=>'ᶠ','g'=>'ᵍ','h'=>'ʰ','i'=>'ⁱ','j'=>'ʲ',
    'k'=>'ᵏ','l'=>'ˡ','m'=>'ᵐ','n'=>'ⁿ','o'=>'ᵒ',
    'p'=>'ᵖ','r'=>'ʳ','s'=>'ˢ','t'=>'ᵗ','u'=>'ᵘ',
    'v'=>'ᵛ','w'=>'ʷ','x'=>'ˣ','y'=>'ʸ','z'=>'ᶻ',
)

# Map every character in a symbol name through the given table.
# Characters not in the table are kept as-is (normal height fallback).
function _map_chars(sym::Symbol, table::Dict{Char,Char})
    join(get(table, c, c) for c in string(sym))
end

# True if the index at position i in a TensorExpression is covariant (lower).
function _is_covariant_idx(idx::TensorIndex)
    haskey(_VBUNDLES, idx.vbundle) && is_down(idx)
end

"""
    _group_index_runs(indices) -> Vector{Tuple{Bool, Vector{Symbol}}}

Group consecutive indices of the same variance into runs.
Each element is `(is_covariant, [sym1, sym2, ...])`.

    [-a1, -a2, a3]  →  [(true,[:a1,:a2]), (false,[:a3])]
"""
function _group_index_runs(indices::Vector{TensorIndex})
    isempty(indices) && return Tuple{Bool,Vector{Symbol}}[]
    runs = Tuple{Bool,Vector{Symbol}}[]
    cur_cov = _is_covariant_idx(indices[1])
    cur_syms = [indices[1].symbol]
    for idx in indices[2:end]
        cov = _is_covariant_idx(idx)
        if cov == cur_cov
            push!(cur_syms, idx.symbol)
        else
            push!(runs, (cur_cov, cur_syms))
            cur_cov  = cov
            cur_syms = [idx.symbol]
        end
    end
    push!(runs, (cur_cov, cur_syms))
    runs
end


# =========================================
# 7.  Formatted output — three shared formatters
# =========================================

"""
    _format_unicode(e::TensorExpression) -> String

Produce a Unicode string suitable for terminal display.

Every character in each index name is mapped to its Unicode sub/superscript
equivalent (letters and digits). Characters without a Unicode sub/sup glyph
are passed through at normal height. No `_` / `^` separator is used — the
visual position itself (subscript/superscript glyph) conveys variance.

Examples:
    g[-a1, -a2]            → "gₐ₁ₐ₂"
    T[a1, -a2]             → "Tᵃ¹ₐ₂"
    R[-a1, -a2, -a3, -a4]  → "Rₐ₁ₐ₂ₐ₃ₐ₄"
    Gamma[a1, -a2]         → "Gammaᵃ¹ₐ₂"
"""
function _format_unicode(e::TensorExpression)
    runs = _group_index_runs(e.indices)
    buf  = string(e.tensor.print_as)
    for (is_cov, syms) in runs
        table = is_cov ? _CHAR_TO_SUB : _CHAR_TO_SUP
        buf  *= join(_map_chars(s, table) for s in syms)
    end
    buf
end

"""
    _format_latex(e::TensorExpression) -> String

Produce a LaTeX math-mode string (without surrounding `\$`).

Examples:
    g[-a1, -a2]       → "g_{a_{1} a_{2}}"
    T[a1, -a2]        → "T^{a_{1}}_{a_{2}}"
"""
function _format_latex(e::TensorExpression)
    # LaTeX: convert trailing digit sequence in a symbol name to _{digit...}.
    function latex_sym(sym::Symbol)
        s = string(sym)
        # Split into leading letters and trailing digits.
        m = match(r"^([^\d]*)(\d+)$", s)
        m === nothing ? s : "$(m[1])_{$(m[2])}"
    end

    runs = _group_index_runs(e.indices)
    buf  = string(e.tensor.print_as)
    for (is_cov, syms) in runs
        body = join(latex_sym(s) * " " for s in syms) |> rstrip
        buf *= is_cov ? "_{$body}" : "^{$body}"
    end
    buf
end

"""
    _format_html(e::TensorExpression) -> String

Produce a styled HTML string for Jupyter / Pluto display.

The tensor name is rendered in an italic serif math font. Consecutive
covariant indices share one `<sub>` tag; consecutive contravariant indices
share one `<sup>` tag. Index symbols are rendered in italic as well,
matching standard mathematical typesetting.

Examples:
    g[-a1, -a2]  → italic g + subscript "a1 a2"
    T[a1, -a2]   → italic T + superscript "a1" + subscript "a2"
"""
function _format_html(e::TensorExpression)
    runs = _group_index_runs(e.indices)
    name = e.tensor.print_as

    # Use a math-style font for the tensor name and index letters.
    # font-style:italic gives the standard italic math appearance;
    # font-family picks a serif math font when available.
    style_name = "style=\"font-style:italic;font-family:'STIX Two Math',serif;\""
    style_idx  = "style=\"font-style:italic;font-size:0.85em;font-family:'STIX Two Math',serif;\""

    buf = "<span $style_name>$name</span>"
    for (is_cov, syms) in runs
        tag   = is_cov ? "sub" : "sup"
        inner = join(("<i>$s</i>" for s in syms), " ")
        buf  *= "<$(tag) $style_idx>$inner</$(tag)>"
    end
    buf
end


# =========================================
# 8.  show methods
# =========================================

"""
    Base.show(io::IO, e::TensorExpression)

Plain-text / REPL display using Unicode sub/superscript digits.

    g[-a1, -a2]            →  gₐ1ₐ2
    R[-a1,-a2,-a3,-a4]     →  Rₐ1ₐ2ₐ3ₐ4
    T[a1, -a2]             →  Tᵃ1ₐ2
"""
function Base.show(io::IO, e::TensorExpression)
    print(io, _format_unicode(e))
end

"""
    Base.show(io::IO, ::MIME"text/latex", e::TensorExpression)

LaTeX display for IJulia / Jupyter notebooks. Renders as typeset math.

    g[-a1, -a2]  →  \$g_{a_{1} a_{2}}\$
    T[a1, -a2]   →  \$T^{a_{1}}_{a_{2}}\$
"""
function Base.show(io::IO, ::MIME"text/latex", e::TensorExpression)
    print(io, "\$", _format_latex(e), "\$")
end

"""
    Base.show(io::IO, ::MIME"text/html", e::TensorExpression)

HTML display for Jupyter / Pluto notebooks.

The tensor name renders in italic serif math font. Consecutive same-variance
indices are grouped into a single `<sub>` or `<sup>` tag, also italic.

IJulia (Jupyter Julia kernel) calls this method automatically when displaying
a `TensorExpression` in a notebook cell output — no explicit `display()` call
needed.

    g[-a1,-a2]    →  g in math font, subscript: a1 a2
    T[a1, -a2]    →  T in math font, superscript: a1, then subscript: a2
"""
function Base.show(io::IO, ::MIME"text/html", e::TensorExpression)
    print(io, _format_html(e))
end


# =========================================
# Exports
# =========================================

export TensorExpression
export tensor_of, indices_of_tensor
# rank_of: already exported from tensors.jl; the TensorExpression method
# is added here via multiple dispatch — no re-export needed.
