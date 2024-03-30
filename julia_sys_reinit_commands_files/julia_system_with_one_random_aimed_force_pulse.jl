y0_lim = 0.25
y_cutoff = 0.1
Ax_lim = 2.0
Ay_lim = 3.0

σx = 0.04 + 0.06 * rand(Float64)
σy = 0.04 + 0.06 * rand(Float64)
x0 = -1.0
y0 = -y0_lim + 2 * y0_lim * rand(Float64)

Ay_max = Ay_lim * (y0 - y_cutoff) / (-y0_lim - y_cutoff)
Ay_min = -Ay_lim * (y0 + y_cutoff) / (y0_lim + y_cutoff)

Ax = Ax_lim * rand(Float64)
Ay = Ay_min + (Ay_max - Ay_min) * rand(Float64)
force_dist = [SpatialGaussian(σx,σy,x0,y0,Ax), SpatialGaussian(σx,σy,x0,y0,Ay)];

function forcing_model!(σ,T,t,fr::AreaRegionCache,phys_params)
    σt = my_params["sigma_1"]
    t0 = my_params["t0_1"]
    modfcn = Gaussian(σt,sqrt(π*σt^2)) >> t0
    σ .= modfcn(t)*fr.generated_field()
end

my_params["sigma_1"] = 0.02
my_params["t0_1"] = 0.5

afm = AreaForcingModel(forcing_model!,spatialfield=force_dist)
forcingdict = Dict("forcing models" => afm);

sys = viscousflow_system(g,bl,phys_params=my_params,bc=bcdict,motions=m,reference_body=1,forcing=forcingdict);
