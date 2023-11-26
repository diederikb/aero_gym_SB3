force_dist_1 = [EmptySpatialField(),SpatialGaussian(
    0.01 + 0.09 *rand(Float64),
    0.01 + 0.09 *rand(Float64),
    -1.05 + 0.1 *rand(Float64),
    -0.05 + 0.1 * rand(Float64),
    -1.0 + 2.0 *rand(Float64))];
force_dist_2 = [EmptySpatialField(),SpatialGaussian(
    0.01 + 0.09 *rand(Float64),
    0.01 + 0.09 *rand(Float64),
    -1.05 + 0.1 *rand(Float64),
    -0.05 + 0.1 * rand(Float64),
    -1.0 + 2.0 *rand(Float64))];

function forcing_model_1!(σ,T,t,fr::AreaRegionCache,phys_params)
    σt = phys_params["sigma_1"]
    t0 = phys_params["t0_1"]
    modfcn = Gaussian(σt,sqrt(π*σt^2)) >> t0
    σ .= modfcn(t)*fr.generated_field()
end

function forcing_model_2!(σ,T,t,fr::AreaRegionCache,phys_params)
    σt = phys_params["sigma_2"]
    t0 = phys_params["t0_2"]
    modfcn = Gaussian(σt,sqrt(π*σt^2)) >> t0
    σ .= modfcn(t)*fr.generated_field()
end

afm_1 = AreaForcingModel(forcing_model_1!,spatialfield=force_dist_1)
afm_2 = AreaForcingModel(forcing_model_2!,spatialfield=force_dist_2)
forcingdict = Dict("forcing models" => [afm_1, afm_2]);

my_params["sigma_1"] = 0.01 + 0.09 * rand(Float64)
my_params["sigma_2"] = 0.01 + 0.09 * rand(Float64)
my_params["t0_1"] = 0.5
my_params["t0_2"] = 0.5 + 1.5 * rand(Float64)
my_params["CFL"] = 0.25 # Because the default timestep_func does not account for point forcing in the flow

sys = viscousflow_system(g,body,phys_params=my_params,bc=bcdict,motions=m,reference_body=1,forcing=forcingdict);
