#!/usr/bin/env julia

module ResEval

export readtodict, sumdict, mindict, argmindict, meanstats

function readfix(filename)
    a = open(filename) do f
        readlines(f)[2:end]
    end
    return a
end

function readtodict(filename, genfirst=false)
    mixed = readfix(filename)
    mixed = map(x -> split(x, ','), mixed)
    mixed = collect(Iterators.flatten(mixed))
    mixed = reshape(mixed, 2, :)

    combined = Array{Union{String, Float64}, 2}(undef, 2, size(mixed, 2))
    combined[1, :] = convert(Array{String, 1}, @view mixed[1, :])
    combined[2, :] = map(x -> parse(Float64, x), @view mixed[2, :])

    dict = Dict{Int16, Dict{Char, Dict{Char, Float64}}}()
    name_re = r"(\d+)-([a-z])-([a-z])-.*"
    for (n, v) in eachcol(combined)
        m = match(name_re, n)
        @assert m !== nothing "$n is a $(typeof(n))"
        imsize = parse(Int16, m.captures[1])
        discnorm, gennorm = map(x -> Char(x[1]), m.captures[2:3])
        if genfirst
            discnorm, gennorm = gennorm, discnorm
        end

        if !haskey(dict, imsize)
            dict[imsize] = Dict{Char, Dict{Char, Float64}}()
        end
        if !haskey(dict[imsize], discnorm)
            dict[imsize][discnorm] = Dict{Char, Float64}()
        end

        dict[imsize][discnorm][gennorm] = v
    end
    return dict
end

function sumdict(dict::Dict{Char, Float64})
    reduce(+, values(dict))
end

function sumdict(dict::Dict)
    mapreduce(sumdict, +, values(dict))
end

function mindict(dict::Dict{Char, Float64})
    minimum(values(dict))
end

function mindict(dict::Dict)
    minimum(map(mindict, values(dict)))
end

function argmindict(dict::Dict{Char, Float64})
    argmin(dict)
end

function argmindict(dict::Dict)
    minargs = Dict(k * ' ' * argmindict(v) => minimum(values(v)) for (k, v) in dict)
    res = argmin(minargs)
    println("min: ", minargs[res], " at ", res)
    return res
end

function getcombmeans(a, b)
    means = Vector{Float64}()
    i = 0
    for (c, d) in zip(a, b)
        cs, cv = split(c, ',')[1:2]
        ds, dv = split(d, ',')[1:2]
        cv = parse(Float64, cv)
        dv = parse(Float64, dv)
        i += 1
        mean = (cv + dv) / 2
        println(i, ' ', cs, "\t\t", mean)
        push!(means, mean)
    end
    println("min: ", minimum(means), '\t', argmin(means))
    println("max: ", maximum(means), '\t', argmax(means))
end

function meanstats()
    a = readfix("ganerator-64-results.csv")
    b = readfix("ganerator-128-results.csv")
    getcombmeans(a, b)
end

end

