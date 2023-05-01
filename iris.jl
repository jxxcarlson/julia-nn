# One layer neural network for the Iris data set

using MLDatasets: Iris
using Flux
using DataFrames
using Random
using ProgressMeter

dataset = Iris()
# > length(dataset)
# > dataset[1:2]


# Extract the data, phase 1
X = transpose(Matrix(dataset[:].features))                                         # size = (4, 150)
y = Matrix(dataset[:].targets)                                                     # size = (150, 1)
y2 = Flux.onehotbatch(y, [ "Iris-setosa",  "Iris-versicolor","Iris-virginica" ])   # size = (3, 150, 1)
y3 = (dropdims(y2, dims=(3)))                                                      # size = (3, 150)

 
# Shuffle the data
Random.seed!(0)
dim = 2
ix = randperm(size(X,dim))
S = mapslices.(x->x[ix], [X,y3], dims=2)
XS = S[1]   # shuffled features
yS = S[2]   # shuffled labels

# Separate the data into training and test sets
X_train, X_test = XS[:, 1:130], XS[:, 130:150]
y_train, y_test = yS[:, 1:130], yS[:, 130:150]
loader = Flux.DataLoader((X_train, y_train))

# Set up a neural network with one 16-node hidden layer
model = Chain(
  Dense(4, 16, relu),
  Dense(16, 3), softmax
)

optim = Flux.setup(Flux.Adam(0.01), model) 

# Train the network, showing progress
losses = []
@showprogress for epoch in 1:100
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end


# Define functions f and g to test the trained model

f(i) = (round.(model(X_test[:,i])), y_test[:,i])

convertBoolVector(b) = Float32[b[i] ? 1.0 : 0.0 for i in 1:length(b)]

g(i) = let 
    a = round.(model(X_test[:,i]))
    b =  convertBoolVector(y_test[:,i])
    return isapprox(a,b)
end
    
# test the model
# 1 means pass, 0 means fail
println("Test model predictions: ", map(g,range(1,21))) 