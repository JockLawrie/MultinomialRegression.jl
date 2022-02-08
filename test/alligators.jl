@testset "Alligators" begin

@info "$(now()) Starting test set: Alligators"

#=
  Stomach contents of alligators in different lakes.
  https://bookdown.org/mpfoley1973/statistics/generalized-linear-models-glm.html#multinomial-logistic-regression
=#

################################################################################
# Supporting functions

function construct_alligator_dataframe(data, xnames, ylevels)
    result = fill("", 219, 5)
    rownum = 0
    for profile = 1:16
        pdata = view(data, :, profile)
        for outcome = 1:5
            nobs = pdata[4 + outcome]
            y    = ylevels[outcome]
            for i = 1:nobs
                rownum += 1
                result[rownum, 1] = y
                for (j, xname) in enumerate(xnames)
                    result[rownum, 1 + j] = pdata[j]
                end
            end
        end
    end
    colnames = vcat("StomachContents", xnames)
    DataFrame(result, colnames)
end

################################################################################
# Data and target parameters (see link above)

ylevels = ["Fish", "Invertebrate", "Reptile", "Bird", "Other"]  # Stomach contents
xnames  = ["profile", "Gender", "Size", "Lake"]                 # Profile of alligators
rawdata = ["1", "f", "<2.3", "george", 3, 9, 1, 0, 1,
"2", "m", "<2.3", "george", 13, 10, 0, 2, 2,
"3", "f", ">2.3", "george", 8, 1, 0, 0, 1,
"4", "m", ">2.3", "george", 9, 0, 0, 1, 2,
"5", "f", "<2.3", "hancock", 16, 3, 2, 2, 3,
"6", "m", "<2.3", "hancock", 7, 1, 0, 0, 5,
"7", "f", ">2.3", "hancock", 3, 0, 1, 2, 3,
"8", "m", ">2.3", "hancock", 4, 0, 0, 1, 2,
"9", "f", "<2.3", "oklawaha", 3, 9, 1, 0, 2,
"10", "m", "<2.3", "oklawaha", 2, 2, 0, 0, 1,
"11", "f", ">2.3", "oklawaha", 0, 1, 0, 1, 0,
"12", "m", ">2.3", "oklawaha", 13, 7, 6, 0, 0,
"13", "f", "<2.3", "trafford", 2, 4, 1, 1, 4,
"14", "m", "<2.3", "trafford", 3, 7, 1, 0, 1,
"15", "f", ">2.3", "trafford", 0, 1, 0, 0, 0,
"16", "m", ">2.3", "trafford", 8, 6, 6, 3, 5]

# Each row is [parameter name, parameter value, stderror]
target = ["(Intercept):1"   -1.8568     0.5813;
"(Intercept):2"   -1.6115     0.5508;
"(Intercept):3"   -2.2866     0.6566;
"(Intercept):4"   -0.6642     0.3802;
"Lakegeorge:1"    -0.5753     0.7952;
"Lakegeorge:2"     1.7805     0.6232;
"Lakegeorge:3"    -1.1295     1.1928;
"Lakegeorge:4"    -0.7666     0.5686;
"Lakeoklawaha:1"  -1.1256     1.1923;
"Lakeoklawaha:2"   2.6937     0.6693;
"Lakeoklawaha:3"   1.4008     0.8105;
"Lakeoklawaha:4"  -0.7405     0.7421;
"Laketrafford:1"   0.6617     0.8461;
"Laketrafford:2"   2.9363     0.6874;
"Laketrafford:3"   1.9316     0.8253;
"Laketrafford:4"   0.7912     0.5879;
"Size>2.3:1"       0.7302     0.6523;
"Size>2.3:2"      -1.3363     0.4112;
"Size>2.3:3"       0.5570     0.6466;
"Size>2.3:4"      -0.2906     0.4599;
"Genderm:1"       -0.6064     0.6888;
"Genderm:2"       -0.4630     0.3955;
"Genderm:3"       -0.6276     0.6853;
"Genderm:4"       -0.2526     0.4663]

################################################################################
# Script

# Dummy-encode categorical data
rawdata        = reshape(rawdata, 9, 16)  # Each column is a vector of data with rownames = vcat(xnames, ylevels)
data           = construct_alligator_dataframe(rawdata, xnames, ylevels)
data.intercept = fill(1.0, nrow(data))
data.male      = [x == "m"        ? 1.0 : 0.0 for x in data.Gender]  # levels = ["f", "m"]
data.large     = [x == ">2.3"     ? 1.0 : 0.0 for x in data.Size]    # levels = ["<2.3", ">2.3"]
data.george    = [x == "george"   ? 1.0 : 0.0 for x in data.Lake]    # levels = ["hancock", "george", "oklawaha", "trafford"]
data.oklawaha  = [x == "oklawaha" ? 1.0 : 0.0 for x in data.Lake]
data.trafford  = [x == "trafford" ? 1.0 : 0.0 for x in data.Lake]

# Construct training data
yname   = "StomachContents"
ylevels = ["Fish", "Bird", "Invertebrate", "Reptile", "Other"]
xnames  = ["intercept", "george", "oklawaha", "trafford", "large", "male"]
ylevels_dict = Dict(ylevel => i for (i, ylevel) in enumerate(ylevels))
y = [ylevels_dict[x] for x in data.StomachContents]
X = Matrix(data[:, xnames])

# Train
model = fit(y, X, yname, ylevels, xnames)

# Assess fitted parameters and standard errors
target_params   = transpose(reshape(target[:, 2], 4, 6))
fitted_params   = coef(model)
target_stderror = transpose(reshape(target[:, 3], 4, 6))
fitted_stderror = stderror(model)
@test maximum(abs.(fitted_params .- target_params)) <= 0.0001
@test maximum(abs.(fitted_stderror .- target_stderror)) <= 0.0001

end