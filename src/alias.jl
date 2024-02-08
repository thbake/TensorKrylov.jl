export KronMat, KronProd, LowerTriangle

# Aliases

# Collection of vectors
const KronProd{T}      = Vector{<:AbstractVector{T}} 

# Matrices represented by (sums of) Kronecker products
const KronMat{matT, I}    = KroneckerMatrix{matT, I}

const KronComp{matT}      = KroneckerMatrixCompact{matT}

# Collection of lower triangular matrices
const LowerTriangle{T} = LowerTriangular{T, <:AbstractMatrix{T}} 

