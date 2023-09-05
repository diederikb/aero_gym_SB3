σx = 0.1 + (2 * rand(Float64) - 1.0) * 0.02
σy = 0.1 + (2 * rand(Float64) - 1.0) * 0.02
x0 = -0.75 + (2 * rand(Float64) - 1.0) * 0.01
y0 = (2 * rand(Float64) - 1.0) * 0.1
amp = (2 * rand(Float64) - 1.0)
force_dist = [EmptySpatialField(),SpatialGaussian(σx,σy,x0,y0,amp)];

function forcing_model!(σ,T,t,fr::AreaRegionCache,phys_params)
    σt = 0.05 + (2 * rand(Float64) - 1.0) * 0.001
    t0 = 0.5
    modfcn = Gaussian(σt,sqrt(π*σt^2)) >> t0
    σ .= modfcn(t)*fr.generated_field()
end

afm = AreaForcingModel(forcing_model!,spatialfield=force_dist)
forcingdict = Dict("forcing models" => afm);

sys = viscousflow_system(g,body,phys_params=my_params,bc=bcdict,motions=m,reference_body=1,forcing=forcingdict);
