module SymbolicTensorsSymbolicsExt

using SymbolicTensors
using Symbolics

SymbolicTensors.is_scalar_like(::Num) = true

SymbolicTensors.scalar_add(a::Num, b::Num) = a + b
SymbolicTensors.scalar_mul(a::Num, b::Num) = a * b
SymbolicTensors.is_scalar_zero(a::Num) = iszero(a)

end
