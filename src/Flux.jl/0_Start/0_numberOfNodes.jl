using Flux, Plots, ProgressMeter;

f(x) = 2x - x^3;

lower = -2;
step = .1f0;
upper = 2;
range = lower:step:upper;
data = [([x], f(x)) for x in range];

n_nodes = [1, 2, 5, 10, 23, 100];
models = [Chain(Dense(1 => n, tanh), Dense(n => 1, bias=false), only) for n in n_nodes];
optims = [Flux.setup(Adam(), model) for model in models];

@showprogress for i in 1:length(n_nodes)
    for epoch in 1:1000
        Flux.train!((m, x, y) -> (m(x) - y)^2, models[i], data, optims[i]);
    end
end

p = plot(x -> f(x), lower, upper, label = "Exact function: f(x) = 2x - x³", title = "Sensitivity to the number of nodes");

for i in eachindex(n_nodes)
    n = n_nodes[i];
    model = models[i];
    scatter!(x -> models[i]([x]), range, label = "Model output, n = $n");
end

display(p);
println("Done. Press a key to exit...");
readline();
