using Random
using LinearAlgebra

# May be used as activation function
sigmoid(x) = 1 ./ (1 .+ exp.(-x))
sigmoid_der(x) = sigmoid(x) .* (1 .- sigmoid(x))

# May be used as loss function
mse(x, y) = sum((x .- y) .^ 2) ./ length(x)
mse_der(x, y) = -2 .* (x .- y) ./ length(x)

mutable struct NeuralNet
    α::Matrix{Float64}
    β::Vector{Float64}
    activation_function::Function
    activation_function_deriv::Function
    loss_function::Function
    loss_function_deriv::Function
    output_function::Function
    output_function_deriv::Function
    hidden_layer_size::Int
    max_rand_weights::Float64
end

"""
Construct a neural network with num_nodes nodes, and max_rand_weights which is
the maximum value a weight can be given when creating the neural net.
"""
function NeuralNet(num_nodes::Int=10, max_rand_weights::Float64=0.1)
    NeuralNet(
        Matrix(undef, 0, 0),
        Vector(undef, 0),
        sigmoid,
        sigmoid_der,
        mse,
        mse_der,
        identity,
        (x -> 1),
        num_nodes,
        max_rand_weights
    )
end

"""
Calculates the Z-matrix, which is the activations of the neurons in the hidden
layer.
"""
function calculate_Z(net::NeuralNet, X::Array)
    Z = zeros(size(X, 1), net.hidden_layer_size)
    for i in 1:size(Z, 1)
        for j in 1:size(Z, 2)
            Z[i, j] = net.activation_function(transpose(net.α[:, j]) * X[i, :])
        end
    end
    Z
end

"""
Calculates the T-matrix, which is the Z-matrix applied with the β-s.
"""
function calculate_T(net::NeuralNet, Z::Array)
    T = zeros(size(Z, 1))
    for i in 1:length(T)
        T[i] = net.β[1] .+ transpose(net.β[2:length(net.β)]) * Z[i, :]
    end
    T
end

"""
Forward-propagation of a neural network given explanatory variables X.
"""
function forward(net::NeuralNet, X::Array)
    Z = calculate_Z(net, X)
    T = calculate_T(net, Z)
    net.output_function.(T)
end

"""
Alias for forward
"""
predict(net::NeuralNet, X::Array) = forward(net, X)

"""
Calculates the loss function of a net with explanatory variables X and response
y, and a regularization parameter λ.
"""
function calculate_loss(net::NeuralNet, X::Array, y::Array, λ::Float64)
    ŷ = forward(net, X)
    net.loss_function(ŷ, y) + λ * (sum(net.α .^ 2) + sum(net.β .^ 2))
end

"""
Calculates the gradient of the loss function with respect to α, for a given net,
response variables y, predicted values ŷ, Z and T-matrices used when gathering
the predictions and the regularization parameter λ.
"""
function calculate_β_grad(net::NeuralNet, y::Array, ŷ::Array, Z::Array, T::Array, λ::Float64)
    β_grad = zeros(net.hidden_layer_size + 1)
    for j in 0:length(β_grad)-1
        for i in 1:size(y, 1)
            if j == 0
                β_grad[1] += -2 * (y[i, :] - ŷ[i, :]) ⋅ net.output_function_deriv.(T[i, :]) + 2 * λ * net.β[1]
            else
                β_grad[j+1] += -2 * (y[i, :] - ŷ[i, :]) ⋅ net.output_function_deriv.(T[i, :]) * Z[i, j] + 2 * λ * net.β[j+1]
            end
        end
    end
    β_grad
end

"""
Calculates the gradient of the loss function with respect to α, for a given net,
response variables y, predicted values ŷ, T-matrix used when gathering the
predictions, the amount of explanatory variables X "p" and the regularization
parameter λ.
"""
function calculate_α_grad(net::NeuralNet, y::Array, ŷ::Array, T::Array, p::Int64, λ::Float64)
    α_grad = zeros(p, net.hidden_layer_size)
    for h in 1:size(α_grad, 1)
        for j in 1:size(α_grad, 2)
            for i in 1:size(y, 1)
                a = net.output_function_deriv(T[i, :]) * net.β[j] * net.activation_function_deriv(sum((net.α[h̃, j] * X[i, h̃] for h̃ in 1:p))) * X[i, h]
                α_grad[h, j] += -2 * (y[i, :] - ŷ[i, :]) ⋅ a + 2 * λ * net.α[h, j]
            end
        end
    end
    α_grad
end

"""
Train a neural network net with explanatory variables given by a matrix X and
response vector y. Additionally one may specify a learning rate γ, the number of
epochs and a regularization parameter (penalty term) λ.
"""
function train(net::NeuralNet, X::Array, y::Array, γ::Float64=0.01, epochs::Int64=100, λ::Float64=0.0)
    size(y, 1) == size(X, 1) || error("y must have the same number of rows as X")

    net.α = 2 * (rand(Float64, size(X, 2), net.hidden_layer_size) .- 0.5) * net.max_rand_weights
    net.β = 2 * (rand(Float64, net.hidden_layer_size + 1) .- 0.5) * net.max_rand_weights

    for epoch in 1:epochs
        loss = calculate_loss(net, X, y)
        println("Epoch $epoch: (loss $loss)")


        Z = calculate_Z(net, X)
        T = calculate_T(net, Z)
        ŷ = forward(net, X) # Can be optimized

        α_grad = calculate_α_grad(net, y, ŷ, T, size(X, 2), λ)
        β_grad = calculate_β_grad(net, y, ŷ, Z, T, λ)

        net.α -= γ * α_grad
        net.β -= γ * β_grad
    end
end


# Demo for neural network
using Plots
Random.seed!(1234)
include("data.jl")
X /= maximum(X)
X_0 = Vector(range(0.0, 1.0, length=101))

net = NeuralNet(10, 0.0001)
train(net, X, y, 0.0001, 5000, 0.0)
ŷ = predict(net, X_0)

scatter(X, y, label="y")
display(plot!(X_0, ŷ[:, 1], label="ŷ"))