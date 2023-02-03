from typing import Final

# XXX: mean, std of XXX
# XXX_B: lower and upper bounds of XXX

''' upper and lower arm scale '''
UA_LENGTH_MEAN: Final[float] = 36.37
UA_LENGTH_STD: Final[float] = 1.82
UA_SCALE: Final[tuple] = (1., UA_LENGTH_STD / UA_LENGTH_MEAN)
UA_SCALE_B: Final[tuple] = (.9, 1.1)

LA_LENGTH_MEAN: Final[float] = 34.9
LA_LENGTH_STD: Final[float] = 1.8
LA_SCALE: Final[tuple] = (1., LA_LENGTH_STD / LA_LENGTH_MEAN)
LA_SCALE_B: Final[tuple] = (.9, 1.1)

''' upper and lower arm mass '''
BODY_MASS_MEAN: Final[float] = 75.
BODY_MASS_STD: Final[float] = 12.

# ratio wrt total body mass
UA_MASS_RATIO: Final[float] = .03
LA_MASS_RATIO: Final[float] = .0175

# total mass of the upper / lower arm
TOT_UA_MASS_MEAN: Final[float] = UA_MASS_RATIO*BODY_MASS_MEAN
TOT_LA_MASS_MEAN: Final[float] = LA_MASS_RATIO*BODY_MASS_MEAN

# mass biases are the masses of the soft colliders
# TODO: Those values have to be updated if mass of soft colliders mass
# are edited: maybe add a test to ensure their correctness?
UA_MASS_BIAS: Final[float] = 0.489
LA_MASS_BIAS: Final[float] = 0.351

# sampled mass = tot_mass - mass_bias
UA_MASS: Final[tuple] = (TOT_UA_MASS_MEAN - UA_MASS_BIAS,
                         UA_MASS_RATIO*BODY_MASS_STD)
LA_MASS: Final[tuple] = (TOT_LA_MASS_MEAN - LA_MASS_BIAS,
                         LA_MASS_RATIO*BODY_MASS_STD)

# mass bounds are determined by non negativity of masses in MJC
MJC_MASS_LB: Final[float] = .001
LA_MASS_B: Final[tuple] = (MJC_MASS_LB, 2*LA_MASS[0] - MJC_MASS_LB)
UA_MASS_B: Final[tuple] = (MJC_MASS_LB, 2*UA_MASS[0] - MJC_MASS_LB)

''' Tissues softness '''
SOLREF_STIFF_B: Final[tuple] = (-975., -100.)
SOLREF_STIFF: Final[tuple] = ((SOLREF_STIFF_B[0]+SOLREF_STIFF_B[1])/2, 150.)
