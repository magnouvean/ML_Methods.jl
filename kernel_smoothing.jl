using LinearAlgebra

struct KernelSmoother
    K::Function
    X::Array
    y::Array
end

"""
Epanechnikov quadratic kernel function with parameter λ
"""
function K_λ(λ)
    if λ ≤ 0
        throw(ArgumentError("λ must be positive"))
    end
    function K(x0, x)
        x̂ = (x - x0) / λ
        x̂² = dot(x̂, x̂)
        x̂² ≤ 1 ? 3 / 4 * (1 - x̂²) : 0
    end
    K
end

"""
Give a kernel smoother based on the training data X, y, and some kernel function
K.
"""
function KernelSmoother(X::Array, y::Array, K::Function=K_λ(1))
    KernelSmoother(K, X, y)
end

function predict(ks::KernelSmoother, X_0::Array)
    ŷ = Matrix(undef, size(X_0, 1), size(X_0, 2))
    for i in 1:size(X_0, 1)
        a = sum((ks.K(X_0[i, :], ks.X[j, :]) * ks.y[j, :]) for j in 1:size(ks.X, 1))
        b = sum((ks.K(X_0[i, :], ks.X[i, :]) for j in 1:size(X, 1)))

        for j in 1:length(a)
            ŷ[i, j] = a[j] / b
        end
    end
    ŷ
end


X = [1.22 0.0; 2.0 0.0; 3.4482 0.0; 4.3221 0.0; 5.000421 0.0; 6.629 0.0; 8.49122 0.0; 10.0491 0.0]
y = [2.4 0.1; 4.0 0.1; 6.6 0.1; 8.4 0.1; 10.0 0.1; 13.2 0.1; 16.8 0.1; 20.0 0.1]

ks = KernelSmoother(X, y, K_λ(10))
ŷ = predict(ks, X)
println(ŷ)

"""
Leave one out cross validation for determining λ in Epanechnikov kernel
"""
function CV_Epanechnikov(λ_list::Vector{Float64}, X::Array, y::Array)
    mse_λ = zeros(length(λ_list))
    for i in eachindex(λ_list)
        mse_cv_sum = 0
        for j in 1:size(X, 1)
            X_cv = X[1:size(X, 1).!=j, :]
            y_cv = y[1:size(y, 1).!=j, :]
            ks = KernelSmoother(X_cv, y_cv, K_λ(λ_list[i]))
            ŷ_cv = predict(ks, X_cv)
            mse_cv_sum += sum((ŷ_cv .- y_cv) .^ 2) / size(y_cv, 1)
        end
        mse_λ[i] = mse_cv_sum / size(X, 1)
    end
    mse_λ
end

λ_list = Vector(range(0.1, 20, step=0.5))
X = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0]
y = 2 / 3 .* X .^ 2 - 4 .* X .+ 6
λ_cvs = CV_Epanechnikov(λ_list, X, y)
println(λ_cvs)

function EpanechnikovCV(X, y, λ_list)
    λ_cv = CV_Epanechnikov(λ_list, X, y)
    _, λ_optim_i = findmin(λ_cv)
    λ_optim = λ_list[λ_optim_i]
    KernelSmoother(X, y, K_λ(λ_optim))
end

ks = EpanechnikovCV(X, y, λ_list)
ŷ = predict(ks, X)
println(y)
println(ŷ)
println("$(sum((ŷ - y).^2) ./ length(y))")