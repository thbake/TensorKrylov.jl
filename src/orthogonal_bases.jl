export arnoldi_modified!, arnoldi_classical!

#function orthonormalize!()

function arnoldi_classical!(A::AbstractMatrix, b::AbstractVector, k::Int, arnoldi::Arnoldi)

    nrm = norm(b) # take the two norm of right hand side 
    
    arnoldi.V[:, 1] = b .* inv(nrm)

    for j = 1:k - 1
    
        w = A * view(arnoldi.V, :, j) # Changes in every iteration of j

        for i = 1:j
        
            arnoldi.H[i, j] = dot(w, @views(arnoldi.V[:, i])) # h_ij
            
        end
    
        arnoldi.V[:, j + 1] = w - (@views(arnoldi.V[:, 1:j]) * @views(arnoldi.H[1:j, j]))
    
        arnoldi.H[j + 1, j] = norm(@views(arnoldi.V[:, j + 1]))
    
        arnoldi.V[:, j + 1] = @views(arnoldi.V[:, j + 1]) .* inv(arnoldi.H[j + 1, j])

    end

end

function arnoldi_modified!(A::AbstractMatrix, b::AbstractVector, k::Int, arnoldi::Arnoldi)

    nrm = norm(b)
    
    arnoldi.V[:, 1] = b .* inv(nrm)

    v = Array{Float64}(undef, (size(A, 1)))
    
    for j = 1:k - 1

        mul!(v, A, @views(arnoldi.V[:, j]))

        for i = 1:j

            arnoldi.H[i, j] = dot(v, @views(arnoldi.V[:, i]))
            
            v .-= arnoldi.H[i, j] * @views(arnoldi.V[:, i])

        end

        arnoldi.H[j + 1, j] = norm(v)

        arnoldi.V[:, j + 1] = v .* inv(arnoldi.H[j + 1, j])

    end

end

