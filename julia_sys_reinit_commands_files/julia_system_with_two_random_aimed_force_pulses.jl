y0_lim = 0.25
y_cutoff = 0.1
Ax_lim = 2.0
Ay_lim = 3.0

σx1 = 0.04 + 0.06 * rand(Float64)
σy1 = 0.04 + 0.06 * rand(Float64)
x01 = -1.0
y01 = -y0_lim + 2 * y0_lim * rand(Float64)

Ay_max = Ay_lim * (y01 - y_cutoff) / (-y0_lim - y_cutoff)
Ay_min = -Ay_lim * (y01 + y_cutoff) / (y0_lim + y_cutoff)

Ax1 = Ax_lim * rand(Float64)
Ay1 = Ay_min + (Ay_max - Ay_min) * rand(Float64)
force_dist_1 = [
    SpatialGaussian(σx1, σy1, x01, y01, Ax1),
    SpatialGaussian(σx1, σy1, x01, y01, Ay1)
];

σx2 = 0.04 + 0.06 * rand(Float64)
σy2 = 0.04 + 0.06 * rand(Float64)
x02 = -1.0
y02 = -y0_lim + 2 * y0_lim * rand(Float64)

Ay_max = Ay_lim * (y02 - y_cutoff) / (-y0_lim - y_cutoff)
Ay_min = -Ay_lim * (y02 + y_cutoff) / (y0_lim + y_cutoff)

Ax2 = Ax_lim * rand(Float64)
Ay2 = Ay_min + (Ay_max - Ay_min) * rand(Float64)
force_dist_2 = [
    SpatialGaussian(σx2, σy2, x02, y02, Ax2),
    SpatialGaussian(σx2, σy2, x02, y02, Ay2)
];

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

my_params["sigma_1"] = 0.02
my_params["t0_1"] = 0.5
my_params["sigma_2"] = 0.02
my_params["t0_2"] = 0.5 + 1.5 * rand(Float64)

afm_1 = AreaForcingModel(forcing_model_1!,spatialfield=force_dist_1)
afm_2 = AreaForcingModel(forcing_model_2!,spatialfield=force_dist_2)
forcingdict = Dict("forcing models" => AbstractForcingModel[afm_1, afm_2]);

sys = viscousflow_system(g,body,phys_params=my_params,bc=bcdict,motions=m,reference_body=1,forcing=forcingdict);
