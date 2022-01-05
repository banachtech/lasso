using CSV, DataFrames, Plots

# Import helper functions
include("GLMnet.jl")

# Prostate data from Tibshirani(1996)
dX = DataFrame(CSV.File("X.csv"))
dy = DataFrame(CSV.File("y.csv"))

X = Matrix(dX)
y = vec(Array(dy))


@time res = GLMnet.elasticnet(X, y, 0.9, maxiter = 100)

