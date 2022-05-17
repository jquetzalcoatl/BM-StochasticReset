using Plots, Interact, SpecialFunctions, Distributions, Statistics, DelimitedFiles
using JSON, Interpolations

γ = 0.57721566490153286061 #Euler Gamma

using QuadGK
# integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
# integral, err = quadgk(x -> 1/√(4π*10)*exp(-(x-2)^2/(4*10))*log(1/√(4π*10)*exp(-(x-2)^2/(4*10))), 0, 1, rtol=1e-8)

N_im(z,z0,d,t) = 1/√(4π*d*t) * (exp(-(z-z0)^2/(4*d*t)) - exp(-(z+z0)^2/(4*d*t))) / erf(z0/√(4*d*t))
ent_N_Im(z,z0,d,t) = -N_im(z,z0,d,t)*log(N_im(z,z0,d,t))

# plot(0:0.1:200, x->ent_N_Im(x,50,1,20))
# integral, err = quadgk(x->ent_N_Im(x,50,1,20), 0, 50+4*√(2*1*20), rtol=1e-8)
# integral, err = quadgk(x->N_im(x,50,1,2), 0, 50+4*√(2*1*20), rtol=1e-8)
# maximum([0,1])
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

plot()
plot(0.2:0.01:15, (w)->abs(tt(2.0,w,1)))
plot(0.2:0.01:25, 1:0.01:5.0, (w,c)->tt2(c,w,1), st=:heatmap, c=cgrad(:matter, 45, rev=true, scale = :exp, categorical=true), xlabel="w", ylabel="c", xscale=:linear)
vline!([exp(-γ/2)], c=:black, lw=3, legend=:none,ls=:dash)
plot(0:0.001:2, w->entPerResCorrected(w,1.1))


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

op = [firstPassage(0.0001, func="ToggleGaussian", x0=[1,2], σ=[1,2]) for i in 1:500]

opp = mean(transpose(hcat([collect(op[i]) for i in 1:10]...)), dims=1)
vcat(hcat([1],opp), hcat([1],opp))

fpstats = simPoints([1.0, 2.7],0.1, 10, z0=[0.1, 2], D=1, Δw=0.5, sampsize=5000, func="Gaussian")

####Evans-Satya model

begin
    fpstats = zeros(1,5) #simPoints([1.8, 2.7],0.1, 10, z0=[0.1, 2], D=1, Δw=0.1, sampsize=50000, func="Gaussian")
    Threads.@threads for δc in 1.8:0.2:3.0 #3.0
        @info δc
        global fpstats = vcat(fpstats,simPoints([δc, 2.7], 0.1, 2,z0=[0.01, 2], D=1, Δw=0.1, sampsize=5000, func="Gaussian"))
    end
    fpstats = fpstats[2:end,:]
end
unique(fpstats[:,end])

begin
    z0 = 0.01
    f = plot()
    for (j,δc) in enumerate(1.2:0.2:3.0)

        f = plot!(0.01:0.01:50, w->2*tt2( δc,w, 1), label=δc, frame=:box, lw=5, s=:auto)
        r = fpstats[fpstats[:, end] .== δc,:]
        # r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== 1.0), :]
        f = plot!(sqrt.([r[i,1] for i in 1:size(r,1)]) .* (z0/(2 * δc )), [r[i,2] for i in 1:size(r,1)] .* 2/ z0^2, seriestype=:scatter, markershapes = :auto, legend=:none, markerstrokewidth=0, ms=10)
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="τ/τ₀", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:linear)
    f = plot(f, ann=[(12.6,0.1,text("c=1.2", 30,:red, font)),(17.97,67,text("c=3.0", 30,:red, font))])
    f = plot!(xlim=(-0.2,22))
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPT_Gauss.png")
exp(-γ/2)


begin
    f = plot()
    for (j,δc) in enumerate(1.4:0.2:2.6)
        # f = plot!(0.1:0.01:10, w-> nmean2(δc,w) + (1 - exp(-w^2)), label=δc, frame=:box, lw=5)
        f = plot!(0.1:0.01:10, w-> nmean2(δc,w), s=:auto, label="c = $(δc)", frame=:box, lw=5, legend=:topleft)
        r = fpstats[fpstats[:, end] .== δc,:]
        f = plot!(sqrt.([r[i,1] for i in 1:size(r,1)]) .* (0.01/(2 * δc)), [r[i,3] for i in 1:size(r,1)],  markershapes = :auto, lw=0, markerstrokewidth=0, ms=8, label=:none)
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash, label=:none)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="μ", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:log)
    f = plot!(ann=[(5,15,text("c=1.2", 30,:red, font)),(2.5,50000,text("c=3.0", 30,:red, font))], legendfontsize=30)
    # f = plot!(legendtitlefonts=30, legendfontsize=30)
    # f = plot!(1:10,x->x+ 2.2^2 + log(√π))
end
begin
    f = plot()
    for (j,δc) in enumerate(1.2:0.2:3.0)
        # f = plot!(0.1:0.01:10, w-> nmean2(δc,w) + (1 - exp(-w^2)), label=δc, frame=:box, lw=5)
        f = plot!(log(0.1):0.01:log(500), u-> log(nmean2(δc,exp(u))), frame=:box, lw=5, legend=(0.81,0.37), s=:auto, label="c = $(δc)")
        r = fpstats[fpstats[:, end] .== δc,:]
        f = plot!(log.(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (0.01/(2 * δc))), [log(r[i,3]) for i in 1:1:size(r,1)], markershapes = :auto, lw=0, label=:none, markerstrokewidth=0, ms=8)
    end
    f = vline!([-γ/2], lw=4, c=:black, ls=:dash, label=:none)
    f = plot!(size=(1200,1500), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ln(μ)", xlabel="ln(w)", guidefontsize=25, tickfont = font(23, "Helvetica"), aspect_ratio=1)
    f = plot(f, ann=[(7.5,8.3,text("c=1.2", 25,:red, font)),(7.5,15.9,text("c=3.0", 25,:red, font)), (8.3,14.1,text("} Δln(μ) = Δc²", 25,:red, font)), (5.4,4,text("1", 25,:black, font)), (6.9,6,text("1", 25,:black, font))], xlim=(-3,11))
    f = plot!(log(50):0.01:log(500),x->x+ 0.6^2 + log(√π), lw=5, c=:black, label=:none)
    f = plot!(log(50):0.01:log(500),x->0.6^2 + log(51) + log(√π), lw=5, c=:black, label=:none)
    f = plot!([log(500), log(500)],[log(50)+ 0.6^2 + log(√π), log(500)+ 0.6^2 + log(√π)], lw=5, c=:black, legendfontsize=30, label=:none)
    # f =
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPNR_Gauss_Log_2.png")


begin
    f = plot()
    for (j,δc) in enumerate(1.8:0.2:3.0)
        f = plot!(0:0.01:10, w-> entPerRes(w)*nmean(δc,w), label=δc, frame=:box, lw=5, s=:auto)
        # f = plot!(0:0.01:10, w-> entPerResCorrected(w,δc)*nmean(δc,w), label=δc, frame=:box, lw=5, s=:auto)
        r = fpstats[fpstats[:, end] .== δc,:]
        f = plot!(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (0.01/(2 * δc)), [r[i,4] for i in 1:1:size(r,1)], seriestype=:scatter, markershape=:auto, label=:none, markerstrokewidth=0, ms=12)
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ΔΣ", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:linear, xscale=:linear, xlim=(0,1), ylim=(-3,1))
    f = plot(f, ann=[(2.6,15,text("c=1.2", 30,:red, font)),(0.97,100,text("c=3.0", 30,:red, font))])
    # plot(f, ann=(0.97,100,text("c=3.0", 25,:red, font)))
    # plot!( -5:8, (-5:8).^2, inset = (1, bbox(0.0,0.1,0.4,0.4)), subplot = 2)
    # plot!( -5:8, 2*(-5:8).^2, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
    # plot!( ff, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPEC_Gauss-testExact-2.png")

writedlm(pwd() * "/MFPT/mfptGauss.dat", hcat([hcat([r[i,j][1] for i in 1:50], [r[i,j][2] for i in 1:50]) for j in 1:15]...))
r = readdlm(pwd() * "/MFPT/mfptGauss.dat")
# r = hcat([[(r[i,2*j-1],r[i,2*j]) for i in 1:50] for j in 1:15]...)
fpstats2 = vcat([hcat(r[:,2*i-1:2*i], zeros(50,2), [1.1 + 0.1*i for j in 1:50]) for i in 1:15]...)

writedlm(pwd() * "/MFPT/mfpnrGauss.dat", hcat([hcat([rr[i,j][1] for i in 1:100], [rr[i,j][2] for i in 1:100]) for j in 1:10]...))
readdlm(pwd() * "/MFPT/mfpnrGauss.dat")

writedlm(pwd() * "/MFPT/mfpecGauss.dat", hcat([hcat([ee[i,j][1] for i in 1:100], [ee[i,j][2] for i in 1:100]) for j in 1:10]...))
readdlm(pwd() * "/MFPT/mfpecGauss.dat")

writedlm(pwd() * "/MFPT/mfpstats1Gauss.dat", fpstats)

writedlm(pwd() * "/MFPT/mfpstats1Gauss-Corrected.dat", fpstats)

fpstats1 = readdlm(pwd() * "/MFPT/mfp2Gauss.dat")
fpstats = readdlm(pwd() * "/MFPT/mfpstats1Gauss.dat")





begin
    f = plot()
    for (j,δc) in enumerate(1.2:0.2:3.0)
        r = fpstats[fpstats[:, end] .== δc,:]
        f = plot!(sqrt.([r[i,1] for i in 1:size(r,1)]) .* (0.1/(2 * δc)), (nmean2.(δc, sqrt.([r[i,1] for i in 1:size(r,1)]) .* (0.1/(2 * δc  ))) .- [r[i,3] for i in 1:size(r,1)]) ./ nmean2.(δc, sqrt.([r[i,1] for i in 1:size(r,1)]) .* (0.1/(2 * δc  ))), markershapes = [:square], lw=1, legend=:none, markerstrokewidth=0, ms=8, frame=:box)
    end
    # plot!()
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="μ Relative error", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:linear, xscale=:linear)
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPNR_Gauss_RelErr.png")

begin
    f = plot(0.2:0.001:1, 0.2:0.001:3.0, (w,c)->entPerRes(w)*nmean2(c,w), st=:heatmap, c=cgrad(:matter, [0.2,0.3,0.3225, 0.3231, 0.4, 0.5], rev=true, scale = :exp, categorical=true), xlabel="w", ylabel="c", xscale=:linear)
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash, label="zero-entropy regime", legend=:topleft)
    f = plot!(size=(1500,1500), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, guidefontsize=25, tickfont = font(23, "Helvetica"), ann=(0.9,2.5,text("ΔΣ", 120, :blue, "Helvetica")), legendfont = font(35, "Helvetica"), yscale=:linear, xscale=:linear)
    # f = plot(f, ann=(0.4,1.5,text("ΔΣ", font(70, "Helvetica", c=:black))))
    # f = plot(f, ann=(0.9,2.5,text("ΔΣ", 120, :blue, "Helvetica")))
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPEC_Gauss_Cont.png")


################
################
################
###TWO GAUSSIANS

Qnormal(c,w) = 1 - 0.5 * exp(-c^2) * (exp(BigFloat((w-c)^2)) * erfc(-c+w) +
            exp(BigFloat((w+c)^2)) * erfc(c+w))

dQnormal(c,w) = -(1/2) * exp(-c^2) * (2*w*(exp((w - c)^2)*erfc(-c + w) + exp((w + c)^2)* erfc(c + w)) - 2*c*(exp((w - c)^2)* erfc(-c + w) - exp((w + c)^2)*erfc(c + w)) - 4/√π)

# Qdelta(y) = 1 - exp(-y)
# timeratio(y,A) = Qnormal(y,A)/Qdelta(y) * (1 - Qdelta(y))/(1-Qnormal(y,A))

tt2Gs(ceven,weven, codd, wodd, z0=1, D=1) = Qnormal(ceven,weven) * (1 + Qnormal(codd,wodd))/(1-Qnormal(codd,wodd) * Qnormal(ceven,weven)) * z0^2/(4*weven^2*ceven^2*D)
tt2Gs0(ceven,weven, codd, wodd, z0=1, D=1) = (1-exp(-2*ceven*weven)) * (1 + Qnormal(codd,wodd))/(1-Qnormal(codd,wodd) * Qnormal(ceven,weven)) * z0^2/(4*weven^2*ceven^2*D)
tt2GsLim(ceven,weven, codd, wodd; z0=1, D=1, z1=1) = weven>5 ? min(tt2Gs(ceven,weven, codd, wodd, z0, D), abs(exp(ceven^2)*weven*√π-1)*2/(1+weven/wodd*exp(-ceven^2 * ((z1/z0 *weven/wodd)^2-1)))*(z0^2/(4*weven^2*ceven^2*D))) : tt2Gs(ceven,weven, codd, wodd, z0, D)
# ttev(c,w, z0=1, D=1) = (exp(2c*w)-1) * z0^2/(4*w^2*c^2*D)

nmean(ceven,weven, codd, wodd) = Qnormal(ceven,weven) * (1 + Qnormal(codd,wodd))/(1-Qnormal(codd,wodd) * Qnormal(ceven,weven)) #mean number of resets
nmeanLim(ceven,weven, codd, wodd; z0=1, D=1, z1=1) = weven>5 ? min(nmean(ceven,weven, codd, wodd), abs(exp(ceven^2)*weven*√π-1)*2/(1+weven/wodd*exp(-ceven^2 * ((z1/z0 *weven/wodd)^2-1)))) : nmean(ceven,weven, codd, wodd)
entPerRes(w) = 0.5*log(w^2 * exp(γ))

totalentropy(ceven,weven, codd, wodd) = (entPerRes(wodd) + entPerRes(weven) * Qnormal(codd,wodd)) * Qnormal(ceven,weven) / (1 - Qnormal(ceven,weven)*Qnormal(codd,wodd))

plot(0.01:0.01:10, weven-> tt2Gs(3,weven,3,weven))

toggleGaus = simPoints([2.0, 2.3],0.1, 10, z0=[0.1, 0.2], D=1, Δw=0.1, sampsize=10000, func="ToggleGaussian")
plot(toggleGaus[:,1], toggleGaus[:,2], st=:scatter)
plot(0.01:0.01:10, weven-> tt2Gs(2.0,weven,2.3,weven) )

hcat(toggleGaus, ones(100,1))
#=
wᵦ = wₐ α
α = zᵦ/zₐ * cₐ/cᵦ = σᵦ/σₐ
cᵦ ∼ cₐ/α   --------- cᵦ=cₐ/α * zᵦ/zₐ
simPoints(c,wmin, wmax; z0=[6],D=1, Δw=0.1, MAXSTEPS=10000000, sampsize=10000, func="Gaussian")
=#




begin
    # @info 1.8, 1
    # fpstats = simPoints([1.8, 1.8], 0.1, 20, z0=[0.1, 0.1], D=1, Δw=0.1, sampsize=50000, func="ToggleGaussian")
    # fpstats = hcat(fpstats, reshape([1.0 for i in 1:size(fpstats,1)], :, 1) )
    fpstats = zeros(1,6)
    for α in 1:0.25:3
        Threads.@threads for δc in 1.6:0.1:4.5
            @info δc, α
            tmp = simPoints([δc, 2*δc/α], 0.1, 1.5, z0=[0.1, 0.2], D=1, Δw=0.1, sampsize=5000, func="ToggleGaussian")
            tmp = hcat(tmp, reshape([α for i in 1:size(tmp,1)], :, 1) )
            global fpstats = vcat(fpstats,tmp)
        end
    end
    fpstats = fpstats[2:end, :]
end
writedlm(pwd() * "/MFPT/mfpstats2Gauss-Corrected.txt", fpstats)

fpstats[(fpstats[:, 5] .== 1.8) .* (fpstats[:, 6] .== 1.0), :]

begin
    z0 = 0.1
    f = plot()
    for α in 1.0:2.0:3.0
        for (j,δc) in enumerate(1.4:0.2:3.0)

            # f = plot!(0.01:0.01:20, w->2*tt2Gs(δc,w, 2*δc/α, α*w), label=δc, frame=:box, lw=5, s=:auto)
            f = plot!(0.01:0.01:20, w->2*tt2GsLim(δc,w, 2*δc/α, α*w, z0=1,z1=2), label=δc, frame=:box, lw=5, s=:auto)
            r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
            f = plot!(sqrt.([r[i,1] for i in 1:5:size(r,1)]) .* (z0/(2 * δc )), [r[i,2] for i in 1:5:size(r,1)] .* 2 / z0^2  , seriestype=:scatter, markershapes = :auto, legend=:none, markerstrokewidth=0, ms=10)
        end
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="τ/τₐ", xlabel="wₐ", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:linear)
    f = plot(f, ann=[(12.6,0.05,text("c=1.8", 30,:red, font)),(30.97,26,text("c=3.0", 30,:red, font))])
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPT_2Gauss.png")
exp(-γ/2)


begin
    z0 = 0.1
    f = plot()
    for α in 1.0:0.5:2.0
        for (j,δc) in enumerate(1.4:0.2:2.6)
            f = plot!(0.1:0.01:10, w-> nmeanLim(δc,w, 2*δc/α, α*w, z0=1,z1=2), s=:auto, label="c = $(δc)", frame=:box, lw=5, legend=:topleft)
            r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
            f = plot!(sqrt.([r[i,1] for i in 1:2:size(r,1)]) .* (z0/(2 * δc)), [r[i,3] for i in 1:2:size(r,1)],  markershapes = :auto, lw=0, markerstrokewidth=0, ms=8, label=:none)
        end
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash, label=:none)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="μ", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:log)
    f = plot!(ann=[(5,15,text("c=1.4", 30,:red, font)),(2.5,50000,text("c=3.0", 30,:red, font))], legendfontsize=30)
    # f = plot!(legendtitlefonts=30, legendfontsize=30)
    # f = plot!(1:10,x->x+ 2.2^2 + log(√π))
end
begin
    z0 = 0.1
    f = plot()
    for α in 1.0:2.0:3.0
        for (j,δc) in enumerate(1.4:0.2:3.0)
            f = plot!(log(0.1):0.01:log(500), u-> log(nmeanLim(δc,exp(u), 2*δc/α,α*exp(u), z0=1,z1=2)), frame=:box, lw=5, legend=:bottomright, s=:auto, label="c = $(δc), α = $(α)")
            r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
            f = plot!(log.(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc))), [log(r[i,3]) for i in 1:1:size(r,1)], markershapes = :auto, lw=0, label=:none, markerstrokewidth=0, ms=8)
        end
    end
    f = vline!([-γ/2], lw=4, c=:black, ls=:dash, label=:none)
    f = plot!(size=(1500,1500), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ln(μ)", xlabel="ln(wₐ)", guidefontsize=25, tickfont = font(23, "Helvetica"), aspect_ratio=1)
    # f = plot(f, ann=[(7.5,8.3,text("c=1.4", 25,:red, font)),(7.5,15.9,text("c=3.0", 25,:red, font)), (8.3,14.1,text("} Δln(μ) = Δc²", 25,:red, font)), (5.4,4,text("1", 25,:black, font)), (6.9,6,text("1", 25,:black, font))])
    f = plot!(f, ann=[(5.4,4,text("1", 25,:black, font)), (6.9,6,text("1", 25,:black, font))], xlim=(-3,15))
    f = plot!(log(50):0.01:log(500),x->x+ 0.6^2 + log(√π), lw=5, c=:black, label=:none)
    f = plot!(log(50):0.01:log(500),x->0.6^2 + log(51) + log(√π), lw=5, c=:black, label=:none)
    f = plot!([log(500), log(500)],[log(50)+ 0.6^2 + log(√π), log(500)+ 0.6^2 + log(√π)], lw=5, c=:black, legendfontsize=30, label=:none)
    # f =
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPNR_2Gauss_Log.png")


begin
    z0 = 0.1
    f = plot()
    for α in 1.0:1.0:3.0
        for (j,δc) in enumerate(1.4:0.2:3.0)
            f = plot!(0:0.01:10, w-> totalentropy(δc, w, 2*δc/α, α*w), label=δc, frame=:box, lw=5, s=:auto)
            r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
            f = plot!(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc)), [r[i,4] for i in 1:1:size(r,1)], seriestype=:scatter, markershape=:auto, legend=:none, markerstrokewidth=0, ms=12)
        end
    end
    f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
    f = hline!([0], lw=4, c=:black, ls=:dash)
    f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ΔΣ", xlabel="wₐ", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:linear, xscale=:linear, xlim=(0,1), ylim=(-8,1))
    f = plot(f, ann=[(2.6,15,text("c=1.2", 30,:red, font)),(0.97,100,text("c=3.0", 30,:red, font))])
    # plot(f, ann=(0.97,100,text("c=3.0", 25,:red, font)))
    # plot!( -5:8, (-5:8).^2, inset = (1, bbox(0.0,0.1,0.4,0.4)), subplot = 2)
    # plot!( -5:8, 2*(-5:8).^2, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
    # plot!( ff, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPEC_2Gauss.png")


fpstats = readdlm(pwd() * "/MFPT/mfpstats2Gauss.txt")
fpstats = readdlm(pwd() * "/MFPT/mfpstats2Gauss-B.txt")




using Roots, Polynomials, Plots

dτ(wa,ca,α) = (dQnormal(ca, wa) + dQnormal(ca, wa) * Qnormal(ca/α, α*wa) + α*Qnormal(ca, wa)*dQnormal(ca/α, α*wa))/(Qnormal(ca, wa)*( 1 + Qnormal(ca/α,α*wa))) + (dQnormal(ca, wa) * Qnormal(ca/α, α*wa) + α*Qnormal(ca, wa)*dQnormal(ca/α, α*wa))/( 1 - Qnormal(ca, wa)*Qnormal(ca/α,α*wa))-2/wa

# s(weven, ceven, α) = totalentropy(ceven,weven, ceven/α, α*weven)
s(weven, ceven, α) = entPerRes(α*weven) + entPerRes(weven) * Qnormal(ceven/α,α*weven)

# plot(0.01:0.01:3, w->s(w, 1.41, 0.202))
function findEntRoot(α, ceven)
    s0(w) = s(w, ceven, α)
    a = find_zero(s0, (0.01,20))
    a
end

function finddtRoot(α, ceven)
    dτ0(w) = dτ(w, ceven, α)
    try
        return fzero(dτ0, 0.1)
    catch
        return 0
    end
end
finddtRoot(1, 2)
findEntRoot(1,1.8)


# plot([findEntRoot(1.2, i) for i in 0.8:0.1:5.8])
# plot!([findEntRoot(0.2, i) for i in 0.8:0.1:5.8])
#
# plot(1:0.01:30, wa->dτ(wa,2.01,0.905))


@manipulate for α in 0.001:0.001:25.0, ceven in 0.01:0.01:20.0, wmax in 0.1:0.1:10, wmin in 0.001:0.001:1
    wevroot = findEntRoot(α, ceven)
    wmindt = finddtRoot(α, ceven)

    f1 = plot(wmin:0.01:wmax, weven -> totalentropy(ceven, weven, ceven/α, α*weven), frame=:box, ylabel="ent", legend=:none)
    f1 = plot!([wevroot], [totalentropy(ceven, wevroot, ceven/α, α*wevroot)], legend=:none, st=:scatter, markerstrokewidth=0)

    f2 = plot(wmin:0.01:wmax, weven-> tt2Gs(ceven, weven, ceven/α, α*weven), frame=:box, lw=5, ylabel="τ")

    # f2 = plot!(wmin:0.01:wmax, weven-> tt2Gs(ceven/α, α*weven, ceven, weven), frame=:box, lw=5, ylabel="τ")

    f2 = plot!([wevroot], [tt2Gs(ceven, wevroot, ceven/α, α*wevroot)], frame=:box, lw=5, ylabel="τ", st=:scatter, markerstrokewidth=0)
    f2 = plot!([α*wevroot], [tt2Gs(ceven, α*wevroot, ceven/α, α*α*wevroot)], frame=:box, lw=5, ylabel="τ", st=:scatter, markerstrokewidth=0)
    f2 = plot!([wmindt], [tt2Gs(ceven, wmindt, ceven/α, α*wmindt)], frame=:box, lw=5, ylabel="τ", st=:scatter, markerstrokewidth=0)

    f2 = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash, label="zero-entropy regime", legend=:none)

    f3 = plot(wmin:0.01:wmax, weven->dτ(weven,ceven,α), legend=:none, xlabel="w", ylabel="dτ/dw")
    f3 = plot!([wmindt], [dτ(ceven, wmindt, α)], frame=:box, lw=5, st=:scatter, markerstrokewidth=0)
    plot(f1,f2, f3, layout= @layout [a ; b ; f3])
end




# ###Not this path...
# α       ceven_min       ceven_max
# 0.7     2.60            10.0
#
# 1.0     1.8
# 1.5     2.32
# 2.0     2.92
# 2.5     3.58
# 3.0     4.26
# 3.5     4.95
# 4.0     5.65
# 4.5     6.33
#
# 2.32/1.5
# 2.92/2
# 3.58/2.5
# 4.26/3
# 4.95/3.5
# 5.65/4
#
# 20/√2
# It appears there's a limit for larg α where the local minimum appears when ceven/α ∼ √2

# {α0, 1.00, 5.0, 1}, {cc, 1.8, 25.0, 0.005}

dd = Dict()
begin
    for α in 0.1:0.1:1.2
        @info α
        α0 = α
        l = [0 0 0 0 0]
        for cc in 1.0:0.0005:10.0
            try
                # @info cc
                dτ0(w) = dτ(w,cc,α0)
                # s0(w) = s(w, cc, α0)
                rs = findEntRoot(α0, cc)

                rt = fzero(dτ0, 0.1)
                l = vcat(l, [α0 cc rs rt  tt2Gs(cc, rt, cc/α0, α0*rt)])
            catch
                nothing
            end
        end
        l = l[2:end, :]
        dd[string(α0)] = l
        #=
            #Given an α and ceven, this stores: α, ceven argmin(entropy) argmin(MFPT) MFPT
        =#
    end
end
dd
####################################
open(pwd() * "/MFPT/dictminMFPTandEnt.json","w") do f
    JSON.print(f, dd)
end
####################################
dict2 = Dict()
open(pwd() * "/MFPT/dictminMFPTandEnt.json", "r") do f
    global dict2
    dict2=JSON.parse(f)  # parse and transform data
end
dd = Dict()
for idx in sort(parse.(Float32, collect(keys(dict2))))
    idx = string(idx)
    dd[idx] = hcat(dict2[idx]...)
end
##############################

sort(parse.(Float32, collect(keys(dd))))[27]
begin
    αmin, αmax = sort(parse.(Float32, collect(keys(dd))))[[1,end]]
    @info αmin, αmax
    f=plot()
    for idx in sort(parse.(Float32, collect(keys(dd))))[1:end]
        idx = string(idx)

        col = (dd[idx][:,5] .- minimum(dd[idx][:,5])) ./ (maximum(dd[idx][:,5])- minimum(dd[idx][:,5])), [parse(Float32, idx)/(αmax+0.1) for i in 1:size(dd[idx][:,5],1)], 1 .- (dd[idx][:,5] .- minimum(dd[idx][:,5])) ./ (maximum(dd[idx][:,5])- minimum(dd[idx][:,5]))

        col = (dd["1.0"][:,5] .- minimum(dd["1.0"][:,5])) ./ (maximum(dd["1.0"][:,5])- minimum(dd["1.0"][:,5])), [parse(Float32, "1.0")/(4+0.1) for i in 1:size(dd["1.0"][:,5],1)], 1 .- (dd["1.0"][:,5] .- minimum(dd["1.0"][:,5])) ./ (maximum(dd["1.0"][:,5])- minimum(dd["1.0"][:,5]))

        f=plot!(dd[idx][:,2], 4 .* dd[idx][:,4] .^2 .* dd[idx][:,2] .^2, markershapes = :circle, lw=10, ms=7, markerstrokewidth=0, yscale=:linear, legend=:none, c= (idx == "1.0" ? :black : :blue))#, label="argmin(MFPT) α=$(idx)" )

        # f=plot!(dd[idx][1:10:end,2], 4 .* dd[idx][1:10:end,4] .^2 .* dd[idx][1:10:end,2] .^2, markershapes = :circle, lw=10, ms=7, markerstrokewidth=0, yscale=:linear, legend=:none, c=RGB.(col[1][1:10:end],col[2][1:10:end],col[3][1:10:end]))

        # f=plot!(dd[idx][:,2], 4 .* dd[idx][:,3] .^2 .* dd[idx][:,2] .^2, lw=5, c=:black)
        #RGB((αmax-parse(Float32, idx))/(0.1+αmax-αmin)), label="zero-entropy α=$(idx)" )
    end
    f=plot!(xaxis="cₐ", yaxis="Q zₐ²/D", leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, frame=:box, size=(1200,700), tickfont = font(20, "Helvetica"), guidefontsize=25 )
    f=plot!([i for i in 1:10], [2.5396 for i in 1:10], label="Evans-Satya model", ls=:dash, lw=5, c=:black)
    f=plot!(ylim=(minimum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)-2, maximum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)+1))
    # f=plot!(ann=[(5.6,7.05,text("Maxwell Demon", 25,:purple, font))], legendfontsize=15)
    f=plot!(ann=[(3.0,2.0,text("↖ Evans-Majumdar", 25,RGB(0.,0.,0.)))], legendfontsize=15)

end
f = plot!(ann=[(1.9,8.2,text("⋆", 60,:orange, font))], legendfontsize=20)
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/Qvsc-2.png")

f = plot!(aa[:,1], aa[:,4], markershapes = :circle, c=:orange, lw=5, ms=3, markerstrokewidth=0)
f = plot!(aa[:,1], aa[:,3], markershapes = :circle, lw=5, c=:red, ms=3, markerstrokewidth=0)
# f = plot!(aa[:,1], aa[:,6], markershapes = :circle, lw=5, c=:orange, ms=3, markerstrokewidth=0)
# f = plot!(ylim=(7,12), xlim=(1,2))
f=plot!(ann=[(5.9,12.5,text("↖ Zero entropy", 30,RGB(0.7,0.2,0), font))], legendfontsize=15)

f=plot!(ann=[(4.5,5.0,text("MFPT Metastability", 25,RGB(1.,1.,1.)))], legendfontsize=15)
# f=plot!(ann=[(9.5,9.4,text("MFPT →", 40,RGB(0.,0.,0.), rotation=-90))], legendfontsize=15)
# f=plot!(ylim=(minimum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)-2, maximum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)+1))

savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/2GaussMinMFPTEnt.png")
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/Qvsc-2.png")

plot!(numericalRootsEntropy[:,end], 4 .* numericalRootsEntropy[:,1] .^ 2 .* numericalRootsEntropy[:,end] .^ 2)


@manipulate for c in 0.1:0.1:4, α in 1:0.1:5
    plot(0:0.1:10, w->√α * exp(-0.5*(γ+Qnormal(c,w))))
    # plot!(0:0.1:10, w->w)
end

f = plot(aa[:,5], aa[:,4], markershapes = :circle, c=:orange, lw=2, ms=3, markerstrokewidth=0)
f = plot!(aa[:,5], aa[:,3], markershapes = :circle, lw=2, c=:red, ms=3, markerstrokewidth=0)
f = plot!(ylim=(7,15))


using Interpolations


function myBisection(f, a,b; steps=10000)
    leftBound, rightBound = a,b
    for st in 1:steps
        if f(a)*f(b)<0
            a = a + (b-a)/2 <= rightBound ? a + (b-a)/2 : a
        elseif f(a)*f(b)>0
            a = a - (b-a) >= leftBound ? a - (b-a) : a
            b = b - (b-a)/2 >= leftBound ? b - (b-a)/2  : b
        elseif f(a)*f(b)==0
            st=steps
        end
    end
    return [a b]
end

function interpolateData(arC, arWMFPT, arWEnt)
    interp_linearMFPT = LinearInterpolation(arC, 4 .* arWMFPT .^2 .* arC .^2)
    interp_linearEnt = LinearInterpolation(arC, 4 .* arWEnt .^2 .* arC .^2)
    fTemp(x) = interp_linearMFPT(x) - interp_linearEnt(x)
    result = myBisection(fTemp, arC[1],arC[end])
    hcat(result, interp_linearMFPT(result[1]), interp_linearEnt(result[2]))
end

# ["cₐ" "cₐ" "reset rate for intersection between entropy and metastable" "reset rate zeroentropy" "α" "MFPT metastable min boundary"]
aa = zeros(1,6)
for idx in sort(parse.(Float32, collect(keys(dd))))
    idx = string(idx)
    @info idx
    aa = vcat(aa, hcat(interpolateData(dd[idx][:,2], dd[idx][:,4], dd[idx][:,3]), parse(Float32, idx), 4 * dd[idx][1,2] ^2 * dd[idx][1,4] ^2))
end
aa = aa[2:end, :]
aa


begin
    f = plot(aa[:,5], sqrt.(aa[:,4] ./ (4 .* aa[:,1] .^2 )) .* exp(γ/2), markershapes = :circle, lw=3, ms=7, markerstrokewidth=0, label="even reset number" )
    plot!(aa[:,5], aa[:,5] .* sqrt.(aa[:,4] ./ (4 .* aa[:,1] .^2 )) .* exp(γ/2),markershapes = :circle, lw=3, ms=7, markerstrokewidth=0, label="odd reset number"  )
    hline!([1], ls=:dash, lw=3, c=:black, label="zero-entropy")
    plot!(xaxis="α", yaxis="exp(Entropy per reset)", leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, frame=:box, size=(1200,700), tickfont = font(20, "Helvetica"), guidefontsize=25, legendfontsize=15, legend=:topleft )
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/2GaussEntPerRes.png")

begin
    f = plot(aa[:,5], aa[:,4], c=:orange, markershapes = :circle, lw=3, ms=10, markerstrokewidth=0, label="zero entropy regime" )
    # f = plot!(aa[:,5], aa[:,3], markershapes = :circle, lw=2, c=:red, ms=7, markerstrokewidth=0, label="MFPT local minimum")
    f = plot!(aa[:,5], aa[:,6], markershapes = :circle, lw=2, c=:auto, ms=0, markerstrokewidth=0, label="MFPT metastable minimum", fillrange=0, fillalpha = 0.35)
    f = plot!(ylim=(7,19), legend=(0.5,0.9), xlabel="α", yaxis="Q zₐ²/D", leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, frame=:box, size=(1200,700), tickfont = font(20, "Helvetica"), guidefontsize=25, legendfontsize=15)
    f=plot!(ann=[(2.8,8.85,text("Maxwell Demon", 25,:purple, font))], legendfontsize=15)
    f=plot!(ann=[(1.6,12.05,text("Externally-Driven \n Regime", 25,:purple))], legendfontsize=15)
    f = plot!(ann=[(1.0,8.2,text("⋆", 60,:orange, font))], legendfontsize=20)
end
savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/2GaussMFPTvsAlpha.png")

#####New section!!!!!!



#Load the fptstats file, then compute the numerical roots for the entropy

begin
    z0 = 0.1
    numericalRootsEntropy = zeros(1, 5)
    for (j,α) in enumerate(unique(fpstats[:,end]))
        for (i,δc) in enumerate(unique(fpstats[:, end-1]))

            # r = fpstats[fpstats[:, end-1] .== δc,:]
            r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
            testF = LinearInterpolation(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc)),[r[i,4] for i in 1:1:size(r,1)])
            result = myBisection(testF, √r[1,1] * (z0/(2 * δc)), √r[end-1,1] * (z0/(2 * δc)))
            @info α, δc, result
            # numericalRootsEntropy[i + (j-1)*180,:] = hcat(result, testF(result[1]), [δc], [α])
            numericalRootsEntropy = vcat(numericalRootsEntropy, hcat(result, testF(result[1]), [δc], [α]))
        end
    end
    numericalRootsEntropy = numericalRootsEntropy[2:end,:]
end
numericalRootsEntropy


begin
    # This is to show if the roots where computed correctly
    z0 = 0.1
    δc = 2.0
    α = 1.0
    # r = fpstats[fpstats[:, end] .== δc,:]
    r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
    nRoots = numericalRootsEntropy[(numericalRootsEntropy[:, end-1] .== δc) .* (numericalRootsEntropy[:, end] .== α), :]
    testF = LinearInterpolation(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc)),[r[i,4] for i in 1:1:size(r,1)])
    plot(sqrt.([r[i,1] for i in 1:1:10]) .* (z0/(2 * δc)),[r[i,4] for i in 1:1:10], st=scatter)
    plot!(0.2:0.001:1, x->testF(x))
    plot!([nRoots[1,1]], [nRoots[1,3]], st=:scatter)
end




##
dd

aaa = numericalRootsEntropy[(numericalRootsEntropy[:, end] .== 2.5), 1]
bbb = numericalRootsEntropy[(numericalRootsEntropy[:, end] .== 2.5), end-1]

sortperm(bbb)
plot(bbb[sortperm(bbb)], 4 .* bbb[sortperm(bbb)] .^ 2 .* aaa[sortperm(bbb)] .^ 2)
plot!(dd["2.5"][:,2], 4 .* dd["2.5"][:,2] .^ 2 .* dd["2.5"][:,4] .^ 2)
LinearInterpolation(bbb[sortperm(bbb)], 4 .* bbb[sortperm(bbb)] .^ 2 .* aaa[sortperm(bbb)] .^ 2 )
interpolateData2(dd["1.0"][:,2], dd["1.0"][:,4], bbb[sortperm(bbb)], aaa[sortperm(bbb)])
##

function interpolateData2(arC, arWMFPT, arCE, arWEnt)
    interp_linearMFPT = LinearInterpolation(arC, 4 .* arWMFPT .^2 .* arC .^2)
    interp_linearEnt = LinearInterpolation(arCE, 4 .* arWEnt .^2 .* arCE .^2)
    fTemp(x) = interp_linearMFPT(x) - interp_linearEnt(x)
    xmin = maximum([arC[1], arCE[1]])
    xmax = minimum([arC[end], arCE[end]])
    result = myBisection(fTemp, xmin,xmax)
    hcat(result, interp_linearMFPT(result[1]), interp_linearEnt(result[2]))
end



# ["cₐ" "cₐ" "reset rate for intersection between entropy and metastable" "reset rate zeroentropy" "α" "MFPT metastable min boundary"]
aa2 = zeros(1,6)
for idx in sort(unique(numericalRootsEntropy[:,end]))
    idxS = string(idx)
    @info idx
    aaa = numericalRootsEntropy[(numericalRootsEntropy[:, end] .== idx), 1]
    bbb = numericalRootsEntropy[(numericalRootsEntropy[:, end] .== idx), end-1]
    aa2 = vcat(aa2, hcat(interpolateData2(dd[idxS][:,2], dd[idxS][:,4], bbb[sortperm(bbb)], aaa[sortperm(bbb)]), idx, 4 * dd[idxS][1,2] ^2 * dd[idxS][1,4] ^2))
end
aa2 = aa2[2:end, :]


begin
    αmin, αmax = sort(parse.(Float32, collect(keys(dd))))[[1,end]]
    @info αmin, αmax
    f=plot()
    for idx in sort(parse.(Float32, collect(keys(dd))))[1:end]
        idx = string(idx)

        col = (dd[idx][:,5] .- minimum(dd[idx][:,5])) ./ (maximum(dd[idx][:,5])- minimum(dd[idx][:,5])), [parse(Float32, idx)/(αmax+0.1) for i in 1:size(dd[idx][:,5],1)], 1 .- (dd[idx][:,5] .- minimum(dd[idx][:,5])) ./ (maximum(dd[idx][:,5])- minimum(dd[idx][:,5]))

        col = (dd["1.0"][:,5] .- minimum(dd["1.0"][:,5])) ./ (maximum(dd["1.0"][:,5])- minimum(dd["1.0"][:,5])), [parse(Float32, "1.0")/(4+0.1) for i in 1:size(dd["1.0"][:,5],1)], 1 .- (dd["1.0"][:,5] .- minimum(dd["1.0"][:,5])) ./ (maximum(dd["1.0"][:,5])- minimum(dd["1.0"][:,5]))

        f=plot!(dd[idx][:,2], 4 .* dd[idx][:,4] .^2 .* dd[idx][:,2] .^2, markershapes = :circle, lw=10, ms=7, markerstrokewidth=0, yscale=:linear, legend=:none, c= (idx == "1.0" ? :black : :blue))#, label="argmin(MFPT) α=$(idx)" )

        # f=plot!(dd[idx][1:10:end,2], 4 .* dd[idx][1:10:end,4] .^2 .* dd[idx][1:10:end,2] .^2, markershapes = :circle, lw=10, ms=7, markerstrokewidth=0, yscale=:linear, legend=:none, c=RGB.(col[1][1:10:end],col[2][1:10:end],col[3][1:10:end]))

        # f=plot!(dd[idx][:,2], 4 .* dd[idx][:,3] .^2 .* dd[idx][:,2] .^2, lw=5, c=:black)
        #RGB((αmax-parse(Float32, idx))/(0.1+αmax-αmin)), label="zero-entropy α=$(idx)" )
    end
    f=plot!(xaxis="cₐ", yaxis="Q zₐ²/D", leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, frame=:box, size=(1200,700), tickfont = font(20, "Helvetica"), guidefontsize=25 )
    f=plot!([i for i in 1:10], [2.5396 for i in 1:10], label="Evans-Satya model", ls=:dash, lw=5, c=:black)
    f=plot!(ylim=(minimum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)-2, maximum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)+1))
    # f=plot!(ann=[(5.6,7.05,text("Maxwell Demon", 25,:purple, font))], legendfontsize=15)
    f=plot!(ann=[(3.0,2.0,text("↖ Evans-Majumdar", 25,RGB(0.,0.,0.)))], legendfontsize=15)

end
f = plot!(ann=[(1.9,8.2,text("⋆", 60,:orange, font))], legendfontsize=20)
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/Qvsc-2.png")

f = plot!(aa[:,1], aa[:,4], markershapes = :circle, c=:orange, lw=5, ms=3, markerstrokewidth=0)
f = plot!(aa[:,1], aa[:,3], markershapes = :circle, lw=5, c=:red, ms=3, markerstrokewidth=0)
# f = plot!(aa[:,1], aa[:,6], markershapes = :circle, lw=5, c=:orange, ms=3, markerstrokewidth=0)
# f = plot!(ylim=(7,12), xlim=(1,2))
f=plot!(ann=[(5.9,12.5,text("↖ Zero entropy", 30,RGB(0.7,0.2,0), font))], legendfontsize=15)

f=plot!(ann=[(4.5,5.0,text("MFPT Metastability", 25,RGB(1.,1.,1.)))], legendfontsize=15)
# f=plot!(ann=[(9.5,9.4,text("MFPT →", 40,RGB(0.,0.,0.), rotation=-90))], legendfontsize=15)
# f=plot!(ylim=(minimum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)-2, maximum(4 .* dd["1.0"][:,4] .^2 .* dd["1.0"][:,2] .^2)+1))

# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/2GaussMinMFPTEnt.png")
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/Qvsc-2.png")

plot!(numericalRootsEntropy[:,end], 4 .* numericalRootsEntropy[:,1] .^ 2 .* numericalRootsEntropy[:,end] .^ 2)

begin
    z0 = 0.1
    # plot()
    for α in 1.0:0.5:3.0

        # r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
        nRoots = numericalRootsEntropy[(numericalRootsEntropy[:, end] .== α), :]
        # testF = LinearInterpolation(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc)),[r[i,4] for i in 1:1:size(r,1)])
        # plot(sqrt.([r[i,1] for i in 1:1:10]) .* (z0/(2 * δc)),[r[i,4] for i in 1:1:10], st=scatter)
        # plot!(0.2:0.001:1, x->testF(x))
        plot!(nRoots[:,end-1], 4 .* nRoots[:,1] .^ 2 .* nRoots[:,end-1] .^ 2, markershapes = :circle, markerstrokewidth=0, label=α)
    end
    plot!()
end
f = plot!(aa2[:,1], aa2[:,3], markershapes = :circle, lw=5, c=:red, ms=3, markerstrokewidth=0)













###################################################################################################
# plot(dd["1.0"][:,2], dd["1.0"][:,3] .* exp(γ/2), ls=:dash, c=:black)
# for idx in sort(parse.(Float32, collect(keys(dd))))[9:end]
#     idxStr = string(idx)
#     plot!(dd[idxStr][:,2], dd[idxStr][:,3] .* exp(γ/2), c=:blue)
#     plot!(dd[idxStr][:,2], idx .* dd[idxStr][:,3] .* exp(γ/2), c=:red)
# end
# plot!()

# ["cₐ" "cₐ" "reset rate for intersection between entropy and metastable" "reset rate zeroentropy" "α" "MFPT metastable min boundary"]
# aa
# aa[10,:]
# [α0 cc rs rt  tt2Gs(cc, rt, cc/α0, α0*rt)]
# dd["1.0"]
# ttt = tt2Gs.(aa[:,1], sqrt.(aa[:,3]) ./ 2 .* aa[:,1], aa[:,1] ./ aa[:,5], aa[:,5] .* .√ aa[:,3] ./2 .* aa[:,1])
# energy(β, σ, t, Q; D=1, z0=1) = β * ( z0^2*(exp(-Q*t)) +(2*D+Q*σ^2)/Q*(1-exp(-Q*t)))
# energy.(1,1 ./ (√2 .* aa[:,1]), ttt, aa[:,3])[5:28]
# plot!(aa[:,5] , energy.(30,1 ./ (√2 .* aa[:,1]), ttt, aa[:,3]))
# plot(0:0.1:1,t->energy(1,1,t,1))

#################2Gauss w/ weight p
Qnormal(c,w) = 1 - 0.5 * exp(-c^2) * (exp(BigFloat((w-c)^2)) * erfc(-c+w) +
            exp(BigFloat((w+c)^2)) * erfc(c+w))
Qdelta(y) = 1 - exp(-y)
timeratio(y,A) = Qnormal(y,A)/Qdelta(y) * (1 - Qdelta(y))/(1-Qnormal(y,A))


Qnormalp(ca,wa, cb,wb, p) = p * Qnormal(ca,wa) +(1-p)*Qnormal(cb,wb)
ttpG(ca,wa, cb,wb, p; z0=6, D=1) = Qnormalp(ca,wa, cb,wb, p)/(1-Qnormalp(ca,wa, cb,wb, p)) * z0^2/(4*wa^2*ca^2*D)
# tt2(c,w,z0=6, D=1) = w>10 ? min(tt(c,w, z0, D), abs(exp(c^2)*w*√π-1)*(z0^2/(4*w^2*c^2*D))) : tt(c,w, z0, D)

nmeanp(ca,wa, cb,wb, p) = Qnormalp(ca,wa, cb,wb, p)/(1-Qnormalp(ca,wa, cb,wb, p)) #mean number of resets
# nmean2(c,w) = w>10 ? min(nmean(c,w), abs(exp(c^2)*w*√π-1)) : nmean(c,w)
entPerResp(wa,wb) = p*0.5*log(wa^2 * exp(γ)) + (1-p)*0.5*log(wb^2 * exp(γ))

plot()
@manipulate for p in 0:0.01:1, ca in 1:0.1:3, α in 1:0.1:3
    plot(0.1:0.01:25, (w)->abs(ttpG(ca,w,ca/α,α*w,p)))
end
plot(0.2:0.01:25, 1:0.01:5.0, (w,c)->tt2(c,w,1), st=:heatmap, c=cgrad(:matter, 45, rev=true, scale = :exp, categorical=true), xlabel="w", ylabel="c", xscale=:linear)
vline!([exp(-γ/2)], c=:black, lw=3, legend=:none,ls=:dash)



#####animations
dt = 0.001

function brownian_motion(n, x0::Float64, F; T=1.0, γ=1.0, Q=0)
  D = T/γ
  x = x0
  traj = [x]
  resPos = [0 0]
  for i in 1:n
    x = x + F(x)*dt/γ + sqrt(2*D*dt)*randn()
    r = rand()
    if r > exp(-Q*dt) && i>10
        x = x0#rand(Normal(2,1))
        resPos = vcat(resPos, [i*dt x])
    else
        nothing
    end
    push!(traj,x)

  end
  return traj, resPos[2:end,:]
end

# function brownian_motion(n, x0::Float64, F, xlim; T=1.0, γ=1.0)
#   D = T/γ
#   x = x0
#   # traj = [x]
#   for i in 1:n
#     x = x + F(x)*dt/γ + sqrt(2*D*dt)*randn()
#     # push!(traj,x)
#     if x >= xlim
#       return i
#     end
#   end
#   return 0
# end
brownian_motion(100,0.1, x->0, T=1, Q=0)
#Free BM

@manipulate for i in 1:1000, Q in 0:0.001:1000
  p1 = plot(legend=:false, title="Brownian Particle w/ Stochastic reset", xlabel="t",
      ylabel="x", frame=:box, tickfont = font(13, "Helvetica"), titlefont = font(13, "Helvetica"))
  # traj = zeros(1, i+1)
  t = collect(0:i) * dt
  traj, resPos = brownian_motion(i,0.1, x->0, T=1.0, Q=Q)
  p1 = plot!(t, traj, lw=3)
  p1 = plot!(resPos[:,1], resPos[:,2], st=:scatter, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=5, c=:red)

  p2 = plot(resPos[:,2], st=:histogram, normalize=true, bins=100, xlabel="Reset Position", ylabel="Histogram", markershapes = :auto, c=:red, label=size(resPos,1), linewidth=0, ms=10, frame=:box)
  p2 = plot!(-5:0.1:5, x->1/√(2π)*exp(-(x-1)^2/(2)), label=:none, lw=4)


  plot(p1,p2, layout = (2,1))
  # plot!(t, reshape(std(traj,dims=1),:), lw=5, c=:red, ls=:dash)
  # hline!([0], lw=0)
end

traj, resPos = brownian_motion(10000,1.0, x->0, T=1.0, Q=10)



@manipulate for i in 1:10000
  p1 = plot(legend=:false, title="Brownian Particle w/ Stochastic reset", xlabel="t",
      ylabel="x", frame=:box, tickfont = font(13, "Helvetica"), titlefont = font(13, "Helvetica"))

  t = collect(0:i-1) * dt

  p1 = plot!(t, traj[1:i], lw=2, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=1)
  p1 = plot!(resPos[resPos[:,1] .< t[end], 1], resPos[resPos[:,1] .< t[end], 2], st=:scatter, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=5, c=:red)

  p2 = plot(-2:0.1:6, x->1/√(2π*(1)^2)*exp(-(x-2)^2/(2*(1)^2)), label=:none, lw=4, c=:black, ls=:dash, xlabel="Reset Position", ylabel="Histogram", frame=:box)
  if size(resPos[resPos[:,1] .< t[end], 1],1) > 0
      p2 = plot!(resPos[resPos[:,1] .< t[end], 2], c=:red, st=:histogram, normalize=true, bins=100, linewidth=0, label=size(resPos[resPos[:,1] .< t[end], 1],1))
  end

  plot(p1,p2, layout = (2,1))
end


g = @animate for i in 1:10000
  p1 = plot(legend=:false, title="Brownian Particle w/ stochastic reset", xlabel="t",
      ylabel="x", frame=:box, tickfont = font(13, "Helvetica"), titlefont = font(13, "Helvetica"))

  t = collect(0:i-1) * dt

  p1 = plot!(t, traj[1:i], lw=2, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=1)
  p1 = plot!(resPos[resPos[:,1] .< t[end], 1], resPos[resPos[:,1] .< t[end], 2], st=:scatter, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=5, c=:red)

  # p2 = plot(-2:0.1:6, x->1/√(2π*(1)^2)*exp(-(x-2)^2/(2*(1)^2)), label=:none, lw=4, c=:black, ls=:dash, xlabel="Reset Position", ylabel="Histogram", frame=:box)
  # if size(resPos[resPos[:,1] .< t[end], 1],1) > 0
  #     p2 = plot!(resPos[resPos[:,1] .< t[end], 2], c=:red, st=:histogram, normalize=true, bins=100, linewidth=0, label=size(resPos[resPos[:,1] .< t[end], 1],1))
  # end

  # plot(p1,p2, layout = (2,1))
end


gif(g, pwd() * "/MFPT/anim_BM_wStochResFixed.gif", fps = 40)

begin
    i=1710
    p1 = plot(legend=:false, xlabel="t",
        ylabel="z", frame=:box, tickfont = font(13, "Helvetica"), titlefont = font(13, "Helvetica"), size=(700,500))

    t = collect(0:i-1) * dt
    tmp = traj[1:i]
    rou = tmp[(traj[1:i] .< 0.01) .* (traj[1:i] .> -0.01)][end]

    p1 = plot!(t, traj[1:i], lw=2, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=1)
    p1 = plot!(resPos[resPos[:,1] .< t[end], 1], resPos[resPos[:,1] .< t[end], 2], st=:scatter, markershapes = :circle, legend=:none, markerstrokewidth=0, ms=5, c=:red)
    p1 = plot!([1.71], [rou], st=:scatter, markershapes = :square, legend=:none, markerstrokewidth=0, ms=5, c=:orange)
end
savefig(p1, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/BMStochRes.png")

begin
    p2 = plot(resPos[resPos[:,1] .< t[end], 2], c=:red, st=:histogram, normalize=false, bins=100, linewidth=0, label="Number of resets = $(size(resPos[resPos[:,1] .< t[end], 1],1))", size=(700,500))
    # p2 = plot!(-2:0.1:6, x->1/√(2π*(1)^2)*exp(-(x-2)^2/(2*(1)^2)), label="Normal Distribution (2,1)", lw=4, c=:black, ls=:dash, xlabel="Reset Position", ylabel="Histogram", frame=:box)
end
savefig(p2, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/resPos.png")


2.53/√2
