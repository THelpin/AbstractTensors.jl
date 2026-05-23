# Micro-benchmark: print_as string compare vs Dict lookup vs direct Int field.
# Included from benchmark/benchmark.jl after setup.

"""
    BenchHead

Lightweight stand-in for a tensor head in sort micro-benchmarks (benchmark 5).
Not used in production — models `registry_id::Int` + `print_as` without changing `Tensor`.
"""
struct BenchHead
    registry_id::Int
    label::String
end

function _bench_heads_from_components(
    components::Vector{TensorComponent},
    tensor_ids::Dict{Any, Int},
)
    out = BenchHead[]
    seen = Set{Any}()
    for c in components
        t = c.tensor
        t in seen && continue
        push!(seen, t)
        push!(out, BenchHead(tensor_ids[t], print_as(t)))
    end
    return sort!(out, by = h -> h.registry_id)
end

function _lex_less_indices(a::Vector{AbstractIndex}, b::Vector{AbstractIndex})
    for (ia, ib) in zip(a, b)
        ia == ib && continue
        return isless(ia, ib)
    end
    return length(a) < length(b)
end

function is_canonical_less_print_as(a::TensorComponent, b::TensorComponent)
    pa = print_as(a.tensor)
    pb = print_as(b.tensor)
    pa != pb && return pa < pb
    return _lex_less_indices(a.indices, b.indices)
end

function is_canonical_less_registry_id(
    a::TensorComponent,
    b::TensorComponent,
    ids::Dict{Any, Int},
)
    a.tensor === b.tensor && return _lex_less_indices(a.indices, b.indices)
    return ids[a.tensor] < ids[b.tensor]
end

function run_sort_benchmark!(components, tensor_ids)
    pairs = Tuple{TensorComponent, TensorComponent}[
        (components[i], components[j])
        for _ in 1:250, i in 1:4, j in 1:4
    ]

    println("\n--- Benchmark 3: is_canonical_less (print_as) ---")
    b_print = @benchmark begin
        for (a, b) in $pairs
            is_canonical_less_print_as(a, b)
        end
    end
    display(b_print)

    println("\n--- Benchmark 4: is_canonical_less (Dict lookup, not field) ---")
    b_id = @benchmark begin
        for (a, b) in $pairs
            is_canonical_less_registry_id(a, b, $tensor_ids)
        end
    end
    display(b_id)

    heads = _bench_heads_from_components(components, tensor_ids)
    head_pairs = Tuple{BenchHead, BenchHead}[
        (heads[i], heads[j])
        for _ in 1:250, i in eachindex(heads), j in eachindex(heads)
    ]
    
    println("\n--- Benchmark 5: head order (String label vs Int registry_id field) ---")
    println("    Mock struct only — models a.tensor.registry_id without changing Tensor.")
    b_label = @benchmark begin
        for (a, b) in $head_pairs
            a.label < b.label
        end
    end
    display(b_label)

    b_field = @benchmark begin
        for (a, b) in $head_pairs
            a.registry_id < b.registry_id
        end
    end
    display(b_field)

    return nothing
end
