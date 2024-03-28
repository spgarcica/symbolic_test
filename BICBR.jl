using SymbolicRegression
using SymbolicRegression: eval_tree_array, Dataset
using DynamicExpressions.EquationModule: Node
using Match


struct Op
    degree::UInt8
    val::Float64
end

struct Prior
    num::UInt64
    op::Dict{String,Op}
end

struct Anl
    unaops::Dict{UInt8,UInt64}
    binops::Dict{UInt8,UInt64}
    cons::UInt64
end

function parse_op(op::AbstractString, val::AbstractString)
    d::String, l::String = split(op, "_")
    deg::UInt8 = 0
    if d == "Nopi" deg = 1
    elseif d == "Nopi2" deg = 2
    else return nothing
    end
    return (l, Op(deg, parse(Float64, val)))
end

function pow2(x) return x^3 end
function pow3(x) return x^3 end

function get_op_str(op)
    hs = hash(op)
    if hs == hash(+) return "+"
    elseif hs == hash(-) return "-"
    elseif hs == hash(*) return "*"
    elseif hs == hash(/) return "/"
    elseif hs == hash(^) return "**"
    elseif hs == hash(safe_pow) return "**"
    elseif hs == hash(sqrt) return "sqrt"
    elseif hs == hash(abs) return "abs"
    elseif hs == hash(sin) return "sin"
    elseif hs == hash(sinh) return "sinh"
    elseif hs == hash(cos) return "cos"
    elseif hs == hash(cosh) return "cosh"
    elseif hs == hash(tan) return "tan"
    elseif hs == hash(tanh) return "tanh"
    elseif hs == hash(exp) return "exp"
    elseif hs == hash(pow2) return "pow2"
    elseif hs == hash(pow3) return "pow3"
    elseif hs == hash(factorial) return "fac"
    end
end

function get_prior(fn::String)
    lines::Vector{String} = readlines(fn)

    ops::Vector{SubString{String}}= split(lines[1])
    vs::Vector{SubString{String}} = split(lines[2])

    return Prior(
        parse(UInt64, vs[1])
        , Dict(map(
            x -> parse_op(x...)
            , zip(ops[2:end], vs[2:end]))))
end
    
function get_ops(n::Node)
    unaops::Dict{UInt8,UInt64} = Dict{UInt8, UInt64}()
    binops::Dict{UInt8,UInt64} = Dict{UInt8, UInt64}()
    cons::Vector{UInt16} = Vector{UInt16}()
    
    for i in n
        if i.degree == 0
            if i.constant
                append!(cons, i.feature)
                continue
            else
                continue
            end
        elseif i.degree == 1
            sel = unaops
        elseif i.degree == 2
            sel = binops
        else
            return nothing
        end
        if haskey(sel, i.op) sel[i.op] += 1 
        else sel[i.op] = 1
        end
    end
    return Anl(unaops, binops, length(Set(cons)))
end

function get_bic(
    tree
    , dataset::Dataset{T,L}
    , options::SymbolicRegression.Options
    , k::Int
)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    sse = sum((prediction .- dataset.y) .^ 2)
    return ((k+1) * log(dataset.n) 
        + (dataset.n * 
            (log(2.0 * pi) 
            + (log(sse) / dataset.n)
            + 1.0)))
end

function get_ener_prior(
    options::SymbolicRegression.Options
    , prior::Prior
    , ops::Anl)
    get_prior = (i, opt) -> prior.op[get_op_str(opt[i])].val

    acc::Float64 = 0.0
    for (k, v) in ops.unaops
        acc += get_prior(k, options.operators.unaops) * v
     end
    for (k, v) in ops.binops
        acc += get_prior(k, options.operators.binops) * v^2
    end
    return acc
end

function get_ener(
    tree
    , dataset::Dataset{T,L}
    , options::SymbolicRegression.Options
    , prior::Prior
)::L where {T,L}
    ops = get_ops(tree)
    println(ops)
    return (
        get_bic(tree, dataset, options, ops.cons) / 2.0
        + get_ener_prior(options, prior, ops))
end

function ener_loss(prior::Prior)
    return (t, d, p) -> get_ener(t, d, p, prior)
end

function load_ds(fn::AbstractString)
    dat = mapreduce(permutedims, vcat
        , map(
            x->map(
                y->parse(Float64,y)
                , split(x))
            , readlines("train.txt")))
    return (transpose(dat[:,1:end-1])
        , transpose(dat[:,end]))
end


X = randn(Float32, 5, 100)
y = 2 * cos.(X[4, :]) + X[1, :] .^ 2 .- 2

prior = get_prior("./prior.dat")
# X, y = load_ds("train.txt")

options = SymbolicRegression.Options(;
    binary_operators=[+, *, /, -]
    , unary_operators=[sin, cos], populations=20
    , loss_function=ener_loss(prior)
)

hall_of_fame = equation_search(
    X, y; niterations=40, options=options, parallelism=:multithreading
)

# dominating = calculate_pareto_frontier(hall_of_fame)

# trees = [member.tree for member in dominating]

# tree = trees[end]
# output, did_succeed = eval_tree_array(tree, X, options)

# println("Complexity\tMSE\tEquation")

# for member in dominating
#     complexity = compute_complexity(member, options)
#     loss = member.loss
#     string = string_tree(member.tree, options)

#     println("$(complexity)\t$(loss)\t$(string)")
# end
