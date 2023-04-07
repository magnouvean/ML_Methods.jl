using Distributions
using Plots
import Random.seed!

seed!(1234)

X = rand(Uniform(0, 1), 200, 1)
y = sin.(4 * X) + rand(Normal(0, 0.1), 200, 1)

p = scatter(X, y)