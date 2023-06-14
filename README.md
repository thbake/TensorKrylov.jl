# TensorKrylov.jl
Solves linear systems of the form

$$\mathbf{A} x = \mathbf{b}$$, 

where 

$$\mathbf{A} = \sum_{s=1}^d I_{n_1} \otimes \cdots \otimes I_{n_{s-1}} \otimes A_s \otimes I_{n_{s+1}} \otimes \cdots \otimes I_{n_d} \in \mathbb{R}^{N \times N}$$,

and 

$$\mathbf{b} = b_1 \otimes \cdots \otimes b_d \in \mathbb{R}^{N \times N}$$

with (tensorized) Krylov subspace methods.
