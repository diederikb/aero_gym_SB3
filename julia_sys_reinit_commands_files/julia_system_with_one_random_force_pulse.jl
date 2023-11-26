σx = 0.01 + 0.09 *rand(Float64)
σy = 0.01 + 0.09 *rand(Float64)
x0 = -1.05 + 0.1 *rand(Float64)
y0 = -0.05 + 0.1 * rand(Float64)
amp = -1.0 + 2.0 *rand(Float64)
force_dist = [EmptySpatialField(),SpatialGaussian(σx,σy,x0,y0,amp)];

function forcing_model!(σ,T,t,fr::AreaRegionCache,phys_params)
    σt = my_params["sigma_1"]
    t0 = my_params["t0_1"]
    modfcn = Gaussian(σt,sqrt(π*σt^2)) >> t0
    σ .= modfcn(t)*fr.generated_field()
end

my_params["sigma_1"] = 0.01 + 0.09 * rand(Float64)
my_params["t0_1"] = 0.5

afm = AreaForcingModel(forcing_model!,spatialfield=force_dist)
forcingdict = Dict("forcing models" => afm);

sys = viscousflow_system(g,body,phys_params=my_params,bc=bcdict,motions=m,reference_body=1,forcing=forcingdict);
