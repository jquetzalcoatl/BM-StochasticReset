include("sim.jl")

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
