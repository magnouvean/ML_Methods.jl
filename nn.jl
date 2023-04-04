using Random

mutable struct NeuralNet
    α::Matrix{Float64}
    β::Matrix{Float64}
    α_bias::Vector{Float64}
    β_bias::Vector{Float64}
    activation_function::Function
    activation_function_deriv::Function
    loss_function::Function
    loss_function_deriv::Function
    output_function::Function
    output_function_deriv::Function
    hidden_layer_size::Int
    max_rand_weights::Float64
end

sigmoid(x) = 1 ./ (1 .+ exp.(-x))
sigmoid_der(x) = sigmoid(x) .* (1 .- sigmoid(x))
mse(x, y) = sum((x .- y) .^ 2) ./ length(x)
mse_der(x, y) = -2 .* (x .- y) ./ length(x)

function NeuralNet(num_nodes::Int=10, max_rand_weights::Float64=0.1)
    NeuralNet(Matrix(undef, 0, 0),
        Matrix(undef, 0, 0),
        Vector(undef, 0),
        Vector(undef, 0),
        sigmoid,
        sigmoid_der,
        mse,
        mse_der,
        identity,
        identity,
        num_nodes,
        max_rand_weights)
end

function calculate_Z(net::NeuralNet, X::Array)
    α_T_X = transpose(net.α) * transpose(X)
    α_bias_matrix = repeat(net.α_bias, 1, size(α_T_X, 2))
    net.activation_function(α_bias_matrix + α_T_X)
end

function calculate_T(net::NeuralNet, Z::Array)
    β_T_Z = transpose(net.β) * Z
    β_bias_matrix = repeat(net.β_bias, 1, size(β_T_Z, 2))
    β_bias_matrix + β_T_Z
end

function forward(net::NeuralNet, X::Array)
    Z = calculate_Z(net, X)
    T = calculate_T(net, Z)
    transpose(net.output_function(T))
end

function calculate_loss(net::NeuralNet, X::Array, y::Array)
    ŷ = forward(net, X)
    net.loss_function(ŷ, y)
end

function train(net::NeuralNet, X::Array, y::Array, γ::Float64=0.01, epochs::Int64=100)
    N = size(X, 1)
    size(y, 1) == N || error("y must have the same number of rows as X")

    p = size(X, 2)
    k = size(y, 2)
    M = net.hidden_layer_size

    net.α = rand(Float64, (p, M)) * net.max_rand_weights
    net.α_bias = rand(Float64, M) * net.max_rand_weights
    net.β = rand(Float64, (M, k)) * net.max_rand_weights
    net.β_bias = rand(Float64, k) * net.max_rand_weights

    for i in 1:epochs
        loss = calculate_loss(net, X, y)
        println("Epoch $i: (loss $loss)")

        α_grad = zeros(p, M)
        α_bias_grad = zeros(M)
        β_grad = zeros(M, k)
        β_bias_grad = zeros(k)
        ŷ = forward(net, X)
        Z = calculate_Z(net, X)
        δ = zeros(k, N)
        s = zeros(M, N)
        for i in 1:N
            z_i = Z[:, i]
            y_i = y[i, :]
            ŷ_i = ŷ[i, :]
            x_i = X[i, :]
            for m in 1:M
                for k_marked in 1:k
                    loss_deriv = net.loss_function_deriv(y_i[k_marked], ŷ_i[k_marked])
                    out_deriv = net.output_function_deriv(transpose(net.β[:, k_marked]) * z_i)
                    δ[k, i] = loss_deriv * out_deriv
                    β_grad[m, k_marked] += δ[k, i] * z_i[m]
                end

                s[m, i] = net.activation_function_deriv(transpose(net.α[:, m]) * x_i) * sum([net.β[m, k] * δ[k, i] for k in 1:k])
                for l in 1:p
                    α_grad[l, m] += s[m, i] * x_i[l]
                end
            end
        end
        net.α = net.α - γ * α_grad
        net.β = net.β - γ * β_grad
    end
end

function predict(net::NeuralNet, X::Array)
    forward(net, X)
end

# Generate some somewhat arbitrary dataset
X = [1.0 0.0; 1.0 1.0; 1.0 1.41; 0.0 2.0; 2.0 1.0; 2.0 1.41; 1.41 2.236; 2.0 2.0; 0.0 3.0; 3.0 1.0] / 3.0
y = [1.0 0.1; 2.0 0.1; 3.0 0.1; 4.0 0.1; 5.0 0.1; 6.0 0.1; 7.0 0.1; 8.0 0.1; 9.0 0.1; 10.0 0.1] / 10.0

# Train the network
net = NeuralNet(20, 0.01)
train(net, X, y, 0.01, 200)
println("Predictions: $(10 * predict(net, X))")