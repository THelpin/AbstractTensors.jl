"""
    expr_form(x)

    Return an expression that reconstructs `x` via its constructor.
    Useful for debugging and development: copy-paste the result to recreate objects.

#### Examples

~~~julia
@def_manifold M 4 [a1, a2, a3, a4] [A1, A2, A3, A4]
expr_form(a1)
# :(CoordinateIndex(:a1, :tangentM))
expr_form(M)
# :(Manifold(...))  # all field values inlined
~~~
"""
function expr_form(x)
    T = typeof(x)

    args = map(fieldnames(T)) do f
        v = getfield(x, f)
        v isa Symbol ? QuoteNode(v) : v
    end

    Expr(:call, GlobalRef(parentmodule(T), nameof(T)), args...)
end

export expr_form