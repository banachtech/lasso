module GLMnet
using LinearAlgebra, Statistics, DataFrames

export elasticnet

function svt(z, α)
    out = 0.0
    if z < -α
        out = (z + α)
    elseif z > α
        out = (z - α)
    end
    return out
end

# Normalize matrix X such that row or column vectors have zero mean and either unit Euclidean length or unit variance
function standardize(X::Matrix{Float64}; isL2 = true, dims = 1)
    Z = fill(0.0, size(X))
    μ = mean(X, dims = dims)
    s = isL2 ? mapslices(norm, X, dims = dims) : std(X, dims = dims, corrected = false)
    Z .= (X .- μ) ./ s
    return Z, vec(μ), vec(s)
end

function colproduct(X::Matrix{Float64})
    m = size(X, 2)
    XX = fill(0.0, m, m)
    for i ∈ 1:m
        for j ∈ i:m
            XX[i, j] = X[:, i]' * X[:, j]
            XX[j, i] = XX[i, j]
        end
    end
    return XX
end

function elasticnet(X::Matrix{Float64}, y::Vector{Float64}, α::Float64; maxiter = 100, K = 100, ϵ = 0.001, labels = nothing, r2tol = 0.01)
    n, m = size(X)
    S = 0.0
    Z, _, _ = standardize(X, isL2 = false, dims = 1)
    ZZ = colproduct(Z) ./ n
    ZY = vec(y' * Z) ./ n
    ỹ = y .- mean(y)
    l2y = sum(abs2, ỹ)
    λmax = maximum(ZY) / α
    λmin = ϵ * λmax
    rng = exp.(range(log(λmax), stop = log(λmin), length = K))
    β = fill(0.0, m)
    βs = fill(0.0, m, K)
    r2 = fill(0.0, K)
    for (k, λ) ∈ enumerate(rng)
        η = λ * α
        γ = (1.0 + λ - η)
        rr = 1.0 - sum(abs2, ỹ .- Z * β) / l2y
        for i ∈ 1:maxiter
            for j ∈ 1:m
                S = ZY[j] - dot(ZZ[j, :], β) + β[j]
                β[j] = svt(S, η) / γ
            end
            tmp = 1.0 - sum(abs2, ỹ .- Z * β) / l2y
            if abs(tmp .- rr) < r2tol
                rr = tmp
                break
            end
        end
        βs[:, k] .= β
        r2[k] = rr
    end
    #identify breakpoints where r2 stops improving
    #r2 = 1.0 .- [sum(abs2, ỹ .- c) for c ∈ eachcol(Z * βs)] ./ sum(abs2, ỹ)
    idx = []
    for l ∈ 2:K
        if abs(r2[l] - r2[l-1]) > r2tol
            push!(idx, l)
        end
    end

    df = m .- vec(mapslices(x -> sum(x .== 0.0), βs, dims = 1)) #number of non-zero coeff
    dfc, λc, r2c = df[idx], rng[idx], r2[idx]
    βc = βs[:, idx]
    l1 = vec(sum(abs, βc, dims = 1)) #l1 norm of solution
    # Collect results into a dataframe
    if isnothing(labels)
        labels = [Symbol("β$l") for l ∈ 1:m]
    end
    res = DataFrame(hcat([dfc λc r2c l1], βc'), [[:df, :λ, :r², :l1]; labels])
    return res
end

end