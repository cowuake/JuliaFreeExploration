using Flux, Measurements, Plots, ProgressMeter, Statistics;

f(x) = 2x - x^3;
#f(x) = x^4 - 3*x^3 -x;

n_trainings = 100;
n_nodes = [1 2 5 10 23 100];
x_lower, x_upper, x_step = -2, 2, .1f0;

x_range = x_lower:x_step:x_upper;
data = [([x], f(x)) for x in x_range];

output = zeros(length(x_range), n_trainings);
plots = [];

@showprogress for n in n_nodes
    for i in 1:n_trainings
        model = Chain(Dense(1 => n, tanh), Dense(n => 1, bias=false), only);
        optim = Flux.setup(Adam(), model);

        for epoch in 1:1000
            Flux.train!((m, x, y) -> (m(x) - y)^2, model, data, optim);
        end

        output[:, i] = [model([x]) for x in x_range];
    end

    p = plot(x -> f(x), x_lower, x_upper, label = "Exact function: f(x) = 2x - x³");
    push!(plots, p);

    average = mean(output, dims = 2);
    stdev = stdm(output, average, dims = 2);
    scatter!(x_range, [average[i] ± stdev[i] for i in 1:length(x_range)], label = "Model output, n = $n");
end

p = plot(plots[1], plots[2], plots[3], plots[4], plots[5], plots[6], layout = (2, 3), plot_title = "Neural network with a single dense layer", size = (1920, 1080));
savefig(p, "plot.png");
gui();
println("Done! Press [ENTER] to exit...");
readline();
