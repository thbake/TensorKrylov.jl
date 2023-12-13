export KronMat, KronProd, LowerTriangle, FMatrices

# Aliases

# Collection of vectors
const KronProd{T}      = Vector{<:AbstractVector{T}} 

# Matrices represented by (sums of) Kronecker products
const KronMat{T}       = KroneckerMatrix{T}

# Collection of lower triangular matrices
const LowerTriangle{T} = LowerTriangular{T, <:AbstractMatrix{T}} 

# Collection of matrices
const FMatrices{T}     = Vector{<:AbstractMatrix{T}} 
