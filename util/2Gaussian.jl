include("sim.jl")
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
        Threads.@threads for δc in 1.6:0.2:2.6
            @info δc, α
            tmp = simPoints([δc, 2*δc/α], 0.1, 1.5, z0=[0.1, 0.2], D=1, Δw=0.1, sampsize=5000, func="ToggleGaussian")
            tmp = hcat(tmp, reshape([α for i in 1:size(tmp,1)], :, 1) )
            global fpstats = vcat(fpstats,tmp)
        end
    end
    fpstats = fpstats[2:end, :]
end
writedlm(pwd() * "/mfpstats2Gauss-Corrected-Large.txt", fpstats)


#
# begin
#     z0 = 0.1
#     f = plot()
#     for α in 1.0:2.0:3.0
#         for (j,δc) in enumerate(1.4:0.2:3.0)
#
#             # f = plot!(0.01:0.01:20, w->2*tt2Gs(δc,w, 2*δc/α, α*w), label=δc, frame=:box, lw=5, s=:auto)
#             f = plot!(0.01:0.01:20, w->2*tt2GsLim(δc,w, 2*δc/α, α*w, z0=1,z1=2), label=δc, frame=:box, lw=5, s=:auto)
#             r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
#             f = plot!(sqrt.([r[i,1] for i in 1:5:size(r,1)]) .* (z0/(2 * δc )), [r[i,2] for i in 1:5:size(r,1)] .* 2 / z0^2  , seriestype=:scatter, markershapes = :auto, legend=:none, markerstrokewidth=0, ms=10)
#         end
#     end
#     f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
#     f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="τ/τₐ", xlabel="wₐ", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:linear)
#     f = plot(f, ann=[(12.6,0.05,text("c=1.8", 30,:red, font)),(30.97,26,text("c=3.0", 30,:red, font))])
# end
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPT_2Gauss.png")



# begin
#     z0 = 0.1
#     f = plot()
#     for α in 1.0:0.5:2.0
#         for (j,δc) in enumerate(1.4:0.2:2.6)
#             f = plot!(0.1:0.01:10, w-> nmeanLim(δc,w, 2*δc/α, α*w, z0=1,z1=2), s=:auto, label="c = $(δc)", frame=:box, lw=5, legend=:topleft)
#             r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
#             f = plot!(sqrt.([r[i,1] for i in 1:2:size(r,1)]) .* (z0/(2 * δc)), [r[i,3] for i in 1:2:size(r,1)],  markershapes = :auto, lw=0, markerstrokewidth=0, ms=8, label=:none)
#         end
#     end
#     f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash, label=:none)
#     f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="μ", xlabel="w", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:log, xscale=:log)
#     f = plot!(ann=[(5,15,text("c=1.4", 30,:red, font)),(2.5,50000,text("c=3.0", 30,:red, font))], legendfontsize=30)
#     # f = plot!(legendtitlefonts=30, legendfontsize=30)
#     # f = plot!(1:10,x->x+ 2.2^2 + log(√π))
# end
# begin
#     z0 = 0.1
#     f = plot()
#     for α in 1.0:2.0:3.0
#         for (j,δc) in enumerate(1.4:0.2:3.0)
#             f = plot!(log(0.1):0.01:log(500), u-> log(nmeanLim(δc,exp(u), 2*δc/α,α*exp(u), z0=1,z1=2)), frame=:box, lw=5, legend=:bottomright, s=:auto, label="c = $(δc), α = $(α)")
#             r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
#             f = plot!(log.(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc))), [log(r[i,3]) for i in 1:1:size(r,1)], markershapes = :auto, lw=0, label=:none, markerstrokewidth=0, ms=8)
#         end
#     end
#     f = vline!([-γ/2], lw=4, c=:black, ls=:dash, label=:none)
#     f = plot!(size=(1500,1500), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ln(μ)", xlabel="ln(wₐ)", guidefontsize=25, tickfont = font(23, "Helvetica"), aspect_ratio=1)
#     # f = plot(f, ann=[(7.5,8.3,text("c=1.4", 25,:red, font)),(7.5,15.9,text("c=3.0", 25,:red, font)), (8.3,14.1,text("} Δln(μ) = Δc²", 25,:red, font)), (5.4,4,text("1", 25,:black, font)), (6.9,6,text("1", 25,:black, font))])
#     f = plot!(f, ann=[(5.4,4,text("1", 25,:black, font)), (6.9,6,text("1", 25,:black, font))], xlim=(-3,15))
#     f = plot!(log(50):0.01:log(500),x->x+ 0.6^2 + log(√π), lw=5, c=:black, label=:none)
#     f = plot!(log(50):0.01:log(500),x->0.6^2 + log(51) + log(√π), lw=5, c=:black, label=:none)
#     f = plot!([log(500), log(500)],[log(50)+ 0.6^2 + log(√π), log(500)+ 0.6^2 + log(√π)], lw=5, c=:black, legendfontsize=30, label=:none)
#     # f =
# end
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPNR_2Gauss_Log.png")


# begin
#     z0 = 0.1
#     f = plot()
#     for α in 1.0:1.0:3.0
#         for (j,δc) in enumerate(1.4:0.2:3.0)
#             f = plot!(0:0.01:10, w-> totalentropy(δc, w, 2*δc/α, α*w), label=δc, frame=:box, lw=5, s=:auto)
#             r = fpstats[(fpstats[:, 5] .== δc) .* (fpstats[:, 6] .== α), :]
#             f = plot!(sqrt.([r[i,1] for i in 1:1:size(r,1)]) .* (z0/(2 * δc)), [r[i,4] for i in 1:1:size(r,1)], seriestype=:scatter, markershape=:auto, legend=:none, markerstrokewidth=0, ms=12)
#         end
#     end
#     f = vline!([exp(-γ/2)], lw=4, c=:black, ls=:dash)
#     f = hline!([0], lw=4, c=:black, ls=:dash)
#     f = plot!(size=(1500,900), leftmargin = 10Plots.mm, bottommargin = 10Plots.mm, ylabel="ΔΣ", xlabel="wₐ", guidefontsize=25, tickfont = font(23, "Helvetica"), yscale=:linear, xscale=:linear, xlim=(0,1), ylim=(-8,1))
#     f = plot(f, ann=[(2.6,15,text("c=1.2", 30,:red, font)),(0.97,100,text("c=3.0", 30,:red, font))])
#     # plot(f, ann=(0.97,100,text("c=3.0", 25,:red, font)))
#     # plot!( -5:8, (-5:8).^2, inset = (1, bbox(0.0,0.1,0.4,0.4)), subplot = 2)
#     # plot!( -5:8, 2*(-5:8).^2, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
#     # plot!( ff, inset = (1, bbox(0.1,0.0,0.4,0.4)), subplot = 2)
# end
# savefig(f, "/Users/javier/Dropbox/Aplicaciones/Overleaf/MFPT/Figs/MFPEC_2Gauss.png")


# fpstats = readdlm(pwd() * "/MFPT/mfpstats2Gauss.txt")
# fpstats = readdlm(pwd() * "/MFPT/mfpstats2Gauss-B.txt")
