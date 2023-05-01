import Pkg; Pkg.add("Plots"); Pkg.add("PyPlot") # or  Pkg.add("PlotlyJS")
using Plots
pyplot()             # or plotlyjs()
plt = plot(sin, -2pi, pi, label="sine function")
display(plt)
