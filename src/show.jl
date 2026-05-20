# Helper: Try HTML display, fall back to text
function render_html(io::IO, x)
    if hasmethod(show, Tuple{IO, MIME"text/html", typeof(x)})
        show(io, MIME"text/html"(), x)
    else
        show(io, x)
    end
end

# For ALL containers (Tuple, AbstractArray)
function Base.show(io::IO, ::MIME"text/html", x::Union{Tuple, AbstractArray})
    open_char = x isa Tuple ? '(' : '['
    close_char = x isa Tuple ? ')' : ']'
    
    print(io, open_char)
    for (i, e) in enumerate(x)
        i > 1 && print(io, ", ")
        render_html(io, e)
    end
    print(io, close_char)
end