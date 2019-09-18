export plot_history


"""`plot_history(h::MVHistory; exclude_keys=[:xrec,])`

Plots the values of each history key over time. If elements in history are
2D arrays only the first axis is plotted ove time.
"""
function plot_history(history::MVHistory; exclude_keys=[:xrec,])
    plt = PyPlot.plt

    plot_keys = collect(keys(history))
    plot_keys = filter!(x -> !(x in exclude_keys), plot_keys)

    N = length(plot_keys)
    fig, axes = plt.subplots(N, 1, sharex=true)
    if N == 1 axes = [axes] end

    for (ax, key) in zip(axes, plot_keys)
        x, y = get(history, key)

        # if dealing with 2D arrays pick just the first axis
        if typeof(y[1]) == Array{eltype(y[1]), 2}
            y = [x[:,1] for x in y]
        end

        if key == :loss
            ax.set_yscale("log")
        end

        ax.plot(x, y)
        ax.set_ylabel(key)
        ax.grid(true)
    end

    plt.tight_layout()
    return fig, axes
end
