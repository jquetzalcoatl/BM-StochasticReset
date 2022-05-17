using Plots, Interact, SpecialFunctions, Distributions, Statistics, DelimitedFiles
using JSON, Interpolations

γ = 0.57721566490153286061 #Euler Gamma

using QuadGK


N_im(z,z0,d,t) = 1/√(4π*d*t) * (exp(-(z-z0)^2/(4*d*t)) - exp(-(z+z0)^2/(4*d*t))) / erf(z0/√(4*d*t))
ent_N_Im(z,z0,d,t) = -N_im(z,z0,d,t)*log(N_im(z,z0,d,t))


function computeEntropyNumerically(z0,d,t; b=1)
    norm, err = quadgk(x->N_im(x,z0,d,t), maximum([0,z0-(10+b)*√(2*d*t)]), z0+b*√(2*d*t), rtol=1e-5)
    if norm > 0.999
        # @info norm, z0, d, t
        ent, err = quadgk(x->ent_N_Im(x,z0,d,t), maximum([0,z0-(10+b)*√(2*d*t)]), z0+b*√(2*d*t), rtol=1e-5)
        return ent
    else
        # @info norm, z0, d, t, b
        ent = computeEntropyNumerically(z0,d,t; b=b+1)
        return ent
    end

end

function getPosition(func, i; z=[4.0, 6.0], σ=[2.0,1.0], p=0.5)
        if func == "Gaussian"
            return rand(Normal(z[1],σ[1]))
        elseif func == "Toggle"
            l = size(z,1)
            return z[i % l + 1]
        elseif func == "Binomial"
            r = rand()
            if r < p
                return z[1]
            else
                return z[2]
            end
        elseif func == "ToggleGaussian"
            l = size(z,1)
            return rand(Normal(z[i % l + 1],σ[i % l + 1]))
        elseif func == "BinomialGaussian"
            r = rand()
            if r < p
                return rand(Normal(z[1],σ[1]))
            else
                return rand(Normal(z[2],σ[2]))
            end
        elseif func == "GaussianN"
            return rand(Normal(z[1],σ[1]/i))
        end
end

function getGaussianEntropy(func, i; σ=[2.0,1.0])
    if func == "Gaussian"
        return σ[1]^2
    elseif func == "ToggleGaussian"
        l = size(σ,1)
        return σ[(i - 1) % l + 1]^2
    end
end

function firstPassage(Q; func="Gaussian", x0=[6], σ=[2.0], MAXSTEPS=10000000, maxResets=0, D=1, p=0.5)
    #=
        this return the first passage time, num of resets and entropy for 1 trajectory. Can fix number of resets to a maximum.
    =#
    totalTime=0
    numResets = 0
    entropy = 0
    tAfterReset = 0

    newposition = getPosition(func, 0; z=x0, σ, p) #x0[1]
    oldposition = newposition
    for i=1:MAXSTEPS
        t_fpt=newposition==0 ? 0 : rand(Levy(0,newposition^2/2*D));  #rand(Levy(0,1/4));      #First passage time Random Variable
        t_reset=-log(rand(Uniform()))/Q            #Transition time random variable
        if t_fpt<t_reset
            totalTime=totalTime+t_fpt
            # numResets=i-1;
            break
        else
            numResets += 1
            newposition =  getPosition(func, i; z=x0, σ=σ, p=p) #rand(Normal(x0,σ))
            totalTime=totalTime+t_reset
            # entropy = entropy + 0.5*log(getGaussianEntropy(func, i; σ) /(2*D*(totalTime-tAfterReset)))
            # integral, err = quadgk(x->-N_im(x,abs(oldposition),D,totalTime-tAfterReset)*log(N_im(x,abs(oldposition),D,totalTime-tAfterReset)), 0, 100, rtol=1e-8)
            integral = computeEntropyNumerically(abs(oldposition),D,totalTime-tAfterReset)
            entropy = entropy + 0.5*(log(2*π*getGaussianEntropy(func, i; σ)) + 1) - integral
            tAfterReset = totalTime
            oldposition = newposition
            if maxResets != 0 && i >= maxResets
                break
            end
        end
        if i==MAXSTEPS
            @warn "increase limit"
        end
    end

    return  totalTime, numResets, entropy
end

function entropyPerReset(Q; x0=6, σ=2.0, maxResets=0, sampSize=20000, MAXSTEPS=1000000)
    #=
        This returns the mean entropy and mean number of resets over "sampsize" trajectories. Can fix the number of resets.
    =#
    res = Array{Tuple{Float64,Int64,Float64}}(undef, 1)
    for i in 1:sampSize
        tmp = firstPassage(Q, x0=x0, σ=σ, maxResets=maxResets, MAXSTEPS=MAXSTEPS)
        if maxResets != 0 && tmp[2] == maxResets
            res = vcat(res, tmp)
        elseif maxResets == 0
            res = vcat(res, tmp)
        end
    end
    popfirst!(res)
    mean([res[i][3] for i in 1:size(res,1)]), mean([res[i][2] for i in 1:size(res,1)])
end

# Theoretic result Entropy 0.5*log(σ^2 * Q * exp(γ)/(2D))
entropyGaus(Q,σ,D) = 0.5*log(σ^2 * Q * exp(γ)/(2*D))
entropyGausCorrected(Q,σ,D,x0) = 0.5*log(σ^2 * Q * exp(γ)/(D*( 2*(1+exp(-x0^2*Q/(4*D))) - π ) ) )

################1 Gauss
Qnormal(c,w) = 1 - 0.5 * exp(-c^2) * (exp(BigFloat((w-c)^2)) * erfc(-c+w) +
            exp(BigFloat((w+c)^2)) * erfc(c+w))
# Qdelta(y) = 1 - exp(-y)
# timeratio(y,A) = Qnormal(y,A)/Qdelta(y) * (1 - Qdelta(y))/(1-Qnormal(y,A))

tt(c,w, z0=6, D=1) = Qnormal(c,w)/(1-Qnormal(c,w)) * z0^2/(4*w^2*c^2*D)
tt2(c,w,z0=6, D=1) = w>10 ? min(tt(c,w, z0, D), abs(exp(c^2)*w*√π-1)*(z0^2/(4*w^2*c^2*D))) : tt(c,w, z0, D)

nmean(c,w) = Qnormal(c,w)/(1-Qnormal(c,w)) #mean number of resets
nmean2(c,w) = w>10 ? min(nmean(c,w), abs(exp(c^2)*w*√π-1)) : nmean(c,w)
entPerRes(w) = 0.5*log(w^2 * exp(γ))
entPerResCorrected(w,c) = -0.5*log(1/(2*w^2) + 1/c^2 +exp(-c^2*w^2)/erf(c*w)*c/(w^2*√π)) + 0.5 * log(π * exp(γ))
entPerResCorrected(w,c) = -0.5*log(1/(2*w^2) +exp(-c^2*w^2)/erf(c*w)*c/(w^2*√π)) + 0.5 * log(π * exp(γ))
entPerResCorrected(w,c) = 0.5*log(2*w^2 * exp(γ)/(4-π))



function simPoints(c,wmin, wmax; z0=[6],D=1, Δw=0.1, MAXSTEPS=100000000, sampsize=10000, func="Gaussian")
    σσ = z0 ./ (√2 .* c)
    # result = [(4*c^2*w^2*D/z0^2, mean([firstPassage(4*c^2*w^2*D/z0^2, x0=z0, σ=σσ, MAXSTEPS=MAXSTEPS, D=D)[idx] for i in 1:sampsize])) for w in wmin:Δw:wmax]
    mfp = [firstPassage(4*c[1]^2*(wmin)^2*D/z0[1]^2, func=func, x0=z0, σ=σσ, MAXSTEPS=MAXSTEPS, D=D) for i in 1:sampsize]
    result = hcat(4*c[1]^2*(wmin)^2*D/z0[1]^2, mean( transpose(hcat([collect(mfp[i]) for i in 1:sampsize]...)), dims=1), c[1])
    # @floop
    # Threads.@threads
    for w in wmin+Δw:Δw:wmax
        mfp = [firstPassage(4*c[1]^2*w^2*D/z0[1]^2, func=func, x0=z0, σ=σσ, MAXSTEPS=MAXSTEPS, D=D) for i in 1:sampsize]
        tmp = hcat(4*c[1]^2*w^2*D/z0[1]^2, mean( transpose(hcat([collect(mfp[i]) for i in 1:sampsize]...)), dims=1), c[1])
        result = vcat(result, tmp)
    end
    result
end
