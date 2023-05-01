# https://github.com/FluxML/model-zoo/tree/master/tutorials/dataloader
# Pkg.add("MLDatasets")

using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch

train_data = MNIST(:train)
# dataset MNIST:
#   metadata    =>    Dict{String, Any} with 3 entries
#   split       =>    :train
#   features    =>    28×28×60000 Array{Float32, 3}
#   targets     =>    60000-element Vector{Int64}

train_x, train_y = train_data[:];
test_x, test_y = MNIST(:test)[:];

train_x = reshape(train_x, 28, 28, 1, :)
test_x = reshape(test_x, 28, 28, 1, :)

train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

println("train_x: ", train_x[1])

data_loader = DataLoader((train_x, train_y), batchsize=128, shuffle=true)

for (x, y) in data_loader
    @assert size(x) == (28, 28, 1, 128) || size(x) == (28, 28, 1, 96)
    @assert size(y) == (10, 128) || size(y) == (10, 96)
    # ...
 end

# println(data_loader)
# DataLoader(::Tuple{Array{Float32, 4}, OneHotArrays.OneHotMatrix{UInt32, Vector{UInt32}}}, shuffle=true, batchsize=128)