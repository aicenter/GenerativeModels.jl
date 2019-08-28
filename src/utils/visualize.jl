export plot_history
export plot_reconstruction
export plot_sine_ode_params


"""`plot_history(h::MVHistory; exclude_keys=[:xrec,])`

Plots the values of each history key over time. If elements in history are
2D arrays only the first axis is plotted ove time.
"""
function plot_history(history::MVHistory; exclude_keys=[:xrec,])
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


"""`plot_reconstruction(model::AbstractGN, X::Array{T,2})

Plots two exemplary reconstructions.
"""
function plot_reconstruction(model::AbstractGN, X::Array{T,2}) where T
    xrec = decoder_mean(model, encoder_mean(model, X)).data

    fig, ax = plt.subplots(1,1)
    
    ax.plot(X, "--", label="Truth", color="gray")
    ax.plot(xrec, label="Rec.", color="C0")
    ax.legend()
    ax.grid()

    return fig, ax
end

function plot_sine_ode_params(constant, specific, U, T, Ω)

    fig, ax = plt.subplots(2,4,figsize=(9,6))
    for a in ax a.grid(true) end

    ax[1,1].plot(Ω, specific[1,:], ".")
    ax[1,1].set_ylabel("s[1,1]")
    ax[1,1].set_title("c[1,1] = $(constant[1])")
    ax[1,1].set_xlabel("ω")

    ax[2,1].plot(Ω, specific[2,:], ".")
    ax[2,1].set_ylabel("s[2,1]")
    ax[2,1].set_title("c[2,1] = $(constant[2])")
    ax[2,1].set_xlabel("ω")

    ax[1,2].plot(Ω, specific[3,:], ".")
    ax[1,2].set_ylabel("s[1,2]")
    ax[1,2].set_title("c[1,2] = $(constant[3])")
    ax[1,2].set_xlabel("ω")

    ax[2,2].plot(Ω, specific[4,:], ".")
    ax[2,2].set_ylabel("s[2,2]")
    ax[2,2].set_title("c[2,2] = $(constant[4])")
    ax[2,2].set_xlabel("ω")

    ax[1,3].plot(Ω, specific[5,:], ".")
    ax[1,3].set_ylabel("s[5]")
    ax[1,3].set_title("c[5] = $(constant[5])")
    ax[1,3].set_xlabel("ω")

    ax[2,3].plot(Ω, specific[6,:], ".")
    ax[2,3].set_ylabel("s[6]")
    ax[2,3].set_title("c[6] = $(constant[6])")
    ax[2,3].set_xlabel("ω")

    ax[1,4].plot(U[1,:], specific[7,:], ".")
    ax[1,4].set_xlabel("xi(t=0)")
    ax[1,4].set_ylabel("s[7]")
    ax[1,4].set_title("c[7] = $(constant[7])")

    ax[2,4].plot(-cos.(T[1,:] .* Ω), specific[8,:], ".")
    ax[2,4].set_xlabel("xi'(t=0)")
    ax[2,4].set_ylabel("s[8]")
    ax[2,4].set_title("c[8] = $(constant[8])")

    for a in ax
        (ymin, ymax) = a.get_ylim()
        if ymax - ymin < 0.1
            a.set_ylim(-1,1)
        end
    end

    plt.tight_layout()
    return fig, ax
end
