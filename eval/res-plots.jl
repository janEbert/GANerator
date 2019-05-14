#!/usr/bin/env julia

module ResPlots

using CSV, DataFrames, Gadfly

export makeboxplots, makegroupedboxplots

function makegroupedboxplots()
    data = CSV.read("results.csv")
    set_default_plot_size(20cm, 20cm)
    p_all_d = plot(data, ygroup="img_shape", x="d_norm", y="FID", color="g_norm", Geom.subplot_grid(Geom.bar(position=:stack)))
    p_all_g = plot(data, ygroup="img_shape", x="d_norm", y="FID", color="g_norm", Geom.subplot_grid(Geom.bar(position=:stack)))
    vstack(p_all_d, p_all_g)
end

function makeboxplots()
    data = CSV.read("results.csv")
    set_default_plot_size(20cm, 20cm)
    p_all_d = stackedboxplot(data, :d_norm, :FID, :g_norm)
    p_all_g = stackedboxplot(data, :g_norm, :FID, :d_norm)
    p_64_d  = stackedboxplot(data[data[:img_shape] .== 64,  :], :d_norm, :FID, :g_norm)
    p_64_g  = stackedboxplot(data[data[:img_shape] .== 64,  :], :g_norm, :FID, :d_norm)
    p_128_d = stackedboxplot(data[data[:img_shape] .== 128, :], :d_norm, :FID, :g_norm)
    p_128_g = stackedboxplot(data[data[:img_shape] .== 128, :], :g_norm, :FID, :d_norm)
    pgrid = gridstack([p_all_d p_all_g; p_64_d p_64_g; p_128_d p_128_g])
    out = SVG("boxplot.svg", 20cm, 20cm)
    draw(out, pgrid)
    return pgrid

    # this is bugged
    #= p_all_stack = title(hstack(p_all_d, p_all_g), "all") =#
    #= p_64_stack  = title(hstack(p_64_d, p_64_g),   "64") =#
    #= p_128_stack = title(hstack(p_128_d, p_128_g), "128") =#
    #= vstack(p_all_stack, p_64_stack, p_128_stack) =#
end

function stackedboxplot(data, x, y, c)
    plot(data, x=x, y=y, color=c, Geom.bar(position=:stack))
end

end
