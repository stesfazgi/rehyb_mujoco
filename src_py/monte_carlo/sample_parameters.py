from matplotlib.pyplot import grid
import numpy as np

from scipy.stats import truncnorm


# TODO: replace constants.py with a json file
from monte_carlo.constants import LA_MASS, LA_MASS_B,  UA_MASS, UA_MASS_B
from monte_carlo.constants import LA_SCALE, LA_SCALE_B, UA_SCALE, UA_SCALE_B
from monte_carlo.constants import SOLREF_STIFF, SOLREF_STIFF_B


''' Normal sampling'''


def get_trunc_gaussian_sample(mean: float, std: float, size: tuple = (1,), lb: float = .9, ub: float = 1.1) -> np.ndarray:
    '''
    A truncated normal distribution is used to prevent generation of extreme scales

    assuming that the default model (ie with scale=1) is an average patient, mean == 1.0
    '''
    if __debug__:
        np.random.seed(1)
    return truncnorm.rvs((lb - mean)/std, (ub - mean)/std, loc=mean, scale=std, size=size)


def get_solref_stiff_sample(samples_nber: tuple = (1.)) -> np.ndarray:
    return np.column_stack((get_trunc_gaussian_sample(*SOLREF_STIFF, samples_nber, *SOLREF_STIFF_B),
                            np.full(samples_nber, -111.)))


KEY_TO_SAMPLER = {
    "scale_ua": lambda samples_nber: get_trunc_gaussian_sample(*UA_SCALE, samples_nber, *UA_SCALE_B),
    "scale_la": lambda samples_nber: get_trunc_gaussian_sample(*LA_SCALE, samples_nber, *LA_SCALE_B),
    "M_ua": lambda samples_nber: get_trunc_gaussian_sample(*UA_MASS, samples_nber, *UA_MASS_B),
    "M_la": lambda samples_nber: get_trunc_gaussian_sample(*LA_MASS, samples_nber, *LA_MASS_B),
    "ua_solrefsmooth": get_solref_stiff_sample,
    "la_solrefsmooth": get_solref_stiff_sample,
}


def generate_mean_samples(samples_nber: int):
    '''
    generates samples all equal to the mean value of each attribute
    '''
    param_samples = {}
    param_samples["scale_ua"] = np.full(samples_nber, UA_SCALE[0])
    param_samples["scale_la"] = np.full(samples_nber, LA_SCALE[0])
    param_samples["M_ua"] = np.full(samples_nber, UA_MASS[0])
    param_samples["M_la"] = np.full(samples_nber, LA_MASS[0])
    # ASK: should we use the LB or the mean??
    # ua_solrefsmooth[i] = [SOLREF_STIFF_LB, -111]
    param_samples["ua_solrefsmooth"] = np.full(
        (samples_nber, 2), [SOLREF_STIFF[0], -111.])
    param_samples["la_solrefsmooth"] = np.full(
        (samples_nber, 2), [SOLREF_STIFF[0], -111.])
    return param_samples


def generate_samples(sampled_params: list, samples_nber: int) -> dict:
    '''
    ! sampled_params must be included in KEY_TO_SAMPLER keys !

    Returned keys: scale_ua, scale_la, M_ua, M_la, ua_solrefsmooth, la_solrefsmooth
    '''
    param_samples = generate_mean_samples(samples_nber)

    for param_name in sampled_params:
        param_samples[param_name] = KEY_TO_SAMPLER[param_name](
            samples_nber)

    return param_samples


def generate_whole_samples(samples_nber: int):
    '''
    This is an optimized version of generate_samples, assuming that we
    want to generate samples for all params
    '''
    param_samples = {}
    param_samples["scale_ua"] = KEY_TO_SAMPLER["scale_ua"](samples_nber)
    param_samples["scale_la"] = KEY_TO_SAMPLER["scale_la"](samples_nber)
    param_samples["M_ua"] = KEY_TO_SAMPLER["M_ua"](samples_nber)
    param_samples["M_la"] = KEY_TO_SAMPLER["M_la"](samples_nber)
    param_samples["ua_solrefsmooth"] = KEY_TO_SAMPLER["ua_solrefsmooth"](
        samples_nber)
    param_samples["la_solrefsmooth"] = KEY_TO_SAMPLER["la_solrefsmooth"](
        samples_nber)

    return param_samples


''' Grid sampling '''


def scale_grid_range(bounds_tuple: tuple, grid_range: np.ndarray) -> np.ndarray:
    return (bounds_tuple[1] - bounds_tuple[0])*grid_range + bounds_tuple[0]


def get_solref_stiff_grid(grid_range: np.ndarray) -> np.ndarray:
    return np.column_stack((scale_grid_range(SOLREF_STIFF_B, grid_range), np.full_like(grid_range, -111.)))


KEY_TO_GRID = {
    "scale_ua": lambda grid_range: scale_grid_range(UA_SCALE_B, grid_range),
    "scale_la": lambda grid_range: scale_grid_range(LA_SCALE_B, grid_range),
    "M_ua": lambda grid_range: scale_grid_range(UA_MASS_B, grid_range),
    "M_la": lambda grid_range: scale_grid_range(LA_MASS_B, grid_range),
    "ua_solrefsmooth": get_solref_stiff_grid,
    "la_solrefsmooth": get_solref_stiff_grid,
}


def generate_grid(sampled_params: list, grid_range: np.ndarray) -> dict:
    '''
    ! sampled_params must be included in KEY_TO_GRID keys !

    Returned keys: scale_ua, scale_la, M_ua, M_la, ua_solrefsmooth, la_solrefsmooth
    '''
    param_grid = generate_mean_samples(len(grid_range))

    # edit desired props
    for param_name in sampled_params:
        param_grid[param_name] = KEY_TO_GRID[param_name](grid_range)

    return param_grid


def generate_whole_grid(grid_range: np.ndarray) -> dict:
    param_samples = {}
    param_samples["scale_ua"] = KEY_TO_GRID["scale_ua"](grid_range)
    param_samples["scale_la"] = KEY_TO_GRID["scale_la"](grid_range)
    param_samples["M_ua"] = KEY_TO_GRID["M_ua"](grid_range)
    param_samples["M_la"] = KEY_TO_GRID["M_la"](grid_range)
    param_samples["ua_solrefsmooth"] = KEY_TO_GRID["ua_solrefsmooth"](
        grid_range)
    param_samples["la_solrefsmooth"] = KEY_TO_GRID["la_solrefsmooth"](
        grid_range)
    return param_samples


if __name__ == "__main__":
    # visual tests
    print("Test 1")
    print(generate_samples(["M_ua", "M_la"], 2))
    print("\nTest 2")
    grid_range = np.linspace(0, 1, 5)
    print(generate_whole_grid(grid_range))
    print("\nTest 3")
    print(get_solref_stiff_grid(grid_range))
    print("\nTest 4")
    print(generate_grid(["scale_ua", "M_la"], grid_range))
