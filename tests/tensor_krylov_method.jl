using TensorKrylov: KroneckerMatrix
using Kronecker

@testset "Compressed residual norm" begin

    H = kroneckersum( repeat( reshape( collect(1:16), 4,4), 4) )
    y = cp_als(rand(4, 4, 4, 4), 3)

end

