'''
Defines Monte Carlo sampler of nesm_with_simple_patient model

The following parameters are sampled:
- upper arm and lower arm scale
- upper arm and lower arm mass
- upper arm and lower arm softness (via Young Modulus)

The sampler relies on a multiprocess implementation
The results are merged in a final unique file

Tip: for server run, use: python -O command to optimize debugs
'''

import multiprocessing as mp
import numpy as np
import os
import csv
import re
import time

from xml.etree import ElementTree as ET
from mujoco_py import load_model_from_path, MjSim
from numpy.random import seed as scipy_seed
from typing import Final
from pickle import load, dump, HIGHEST_PROTOCOL

from shared_utils.general import get_project_root, remove_tmp_files, models_folder_path, gen_models_folder_path
from shared_utils.xacro import set_xacro_property, parse_xacro_to_xml, register_xacro_namespace
from xml_generation.utils import save_xml_file
from soft_tissues_eval.utils import edit_MjModel_cmp_smoothness
from monte_carlo.sample_parameters import generate_samples, generate_grid
from monte_carlo.experiment_protocols import spasticity_assessment, StaticData


''' Paths '''
# set path to reference xacro model that will be sampled
XACRO_REL_PATH: Final[str] = os.path.relpath(os.path.join(
    models_folder_path(), "exo_with_patient", "nesm_with_simple_patient.xacro"))

# set path to script tmp folder
# relpath from project root
ROOT_TO_TMP: Final[str] = "bin/outputs/mc_tmp"
PATH_TO_TMP: Final[str] = os.path.join(get_project_root(), ROOT_TO_TMP)
# set path to results folder
PATH_TO_RES: Final[str] = os.path.join(
    get_project_root(), "bin", "outputs", "mc_results")


''' Sampling keys'''
# experience outputs names
EXP_OUTPUTS: Final[list] = [
    "estimated_stiffness", "rough_ground_truth_stiffness", "max_contact_force"]
# experience flags names
EXP_FLAGS: Final[list] = ["experiment_flag", "initialization_flag"]
# sampled xacro props
XACRO_PROPS: Final[list] = ["scale_ua", "scale_la", "M_ua", "M_la"]
# sampled solref props
SOLREF_PROPS: Final[list] = ["ua_solrefsmooth", "la_solrefsmooth"]

SAMPLING_KEYS: Final[set] = set(XACRO_PROPS+SOLREF_PROPS)
CSV_HEADER: Final[list] = XACRO_PROPS + SOLREF_PROPS + \
    EXP_OUTPUTS + EXP_FLAGS + ["Process_name"]

RELOAD_STATIC_DATA: Final[bool] = False


class SamplingRules():
    def __init__(self, use_grid: bool = False, sampled_props: list = []):
        '''
        use_grid activates grid mode in sampling (otherwise gaussian sampling)
        sampled_pops are props that will be sampled, the others will be set to mean
        '''
        self.use_grid = use_grid

        assert all(
            [prop in SAMPLING_KEYS for prop in sampled_props]), "Unknown prop"
        self.sampled_props = sampled_props


def load_static_data_backup(force_reload: bool = False) -> StaticData:
    data_backup_path = os.path.join(
        get_project_root(), "bin", "outputs", "mc_params", "static_data_backup.pkl")

    if os.path.isfile(data_backup_path) and not force_reload:
        with open(data_backup_path, 'rb') as inp:
            static_data = load(inp)
    else:
        static_model_path = os.path.join(gen_models_folder_path(),
                                         "exo_with_patient",
                                         "nesm_with_simple_patient.xml")

        if not os.path.isfile(static_model_path):
            print("You must generate the model once before starting the script!")
            exit(1)
        static_data = StaticData(load_model_from_path(static_model_path))
        with open(data_backup_path, 'wb') as outp:
            dump(static_data, outp, HIGHEST_PROTOCOL)

    return static_data


# nominal_model, sim_nominal,
def evaluate_n_samples(sampling_rules: SamplingRules, static_data: StaticData, samples_nber: int = 1, grid_range: np.ndarray = None):
    '''
    Evaluate 'samples_nber' samples on one process
    '''
    # import nominal model
    nominal_model_path = os.path.join(
        models_folder_path(), "exo_with_patient", "fully_connected_simple.xml")
    assert os.path.isfile(nominal_model_path)
    nominal_model = load_model_from_path(nominal_model_path)

    sim_nominal = MjSim(nominal_model)  # nominal_model, sim_nominal,

    # load original xacro model
    xacro_root = ET.parse(XACRO_REL_PATH).getroot()

    # the gen xml model is not in the usual location 'bin/models/exo_with_patient'
    # Therefore the 'path_to_root' property has to be adjusted
    set_xacro_property(xacro_root, "path_to_root", "../../..")

    # set new seed for scipy (useful for multiproc)
    scipy_seed()

    # there is one tmp filename by process
    thread_id = mp.current_process().name
    edited_xacro_filename = f"nesm_with_simple_patient_tmp_{thread_id}.xacro"
    xml_filename = f"nesm_with_simple_patient_tmp_{thread_id}.xml"

    if sampling_rules.use_grid:
        param_samples = generate_grid(
            sampling_rules.sampled_props, grid_range)

    else:
        param_samples = generate_samples(
            sampling_rules.sampled_props, samples_nber)

    # generate map for assessment exit()results
    assessment_results = {}
    assessment_results["outputs"] = np.zeros((samples_nber, len(EXP_OUTPUTS)))
    assessment_results["flags"] = np.zeros(
        (samples_nber, len(EXP_FLAGS)), dtype=np.bool_)

    for idx in range(samples_nber):
        # set properties in the original xacro_model
        for prop in XACRO_PROPS:
            set_xacro_property(xacro_root, prop, str(param_samples[prop][idx]))

        # save the new xacro file in tmp
        save_xml_file(xacro_root, edited_xacro_filename, PATH_TO_TMP)

        # parse newly generated xacro file into xml (uses xacro.sh)
        parse_xacro_to_xml(edited_xacro_filename,
                           xml_filename, ROOT_TO_TMP)

        # load the model
        try:
            model = load_model_from_path(
                os.path.join(PATH_TO_TMP, xml_filename))
        except Exception as e:
            print(e)

        # adjust upper and lower arm softness
        edit_MjModel_cmp_smoothness(
            model, "ua", param_samples["ua_solrefsmooth"][idx]
        )
        edit_MjModel_cmp_smoothness(
            model, "la", param_samples["la_solrefsmooth"][idx]
        )

        # start spasticity assessment
        outputs, flags = spasticity_assessment(model, sim_nominal, static_data)

        # save sample results in an array
        assessment_results["outputs"][idx] = outputs
        assessment_results["flags"][idx] = flags

    # save results to csv file (assume unique thread name)
    with open(os.path.join(PATH_TO_RES, f"results_tmp_{thread_id}.csv"), 'a') as f:
        writer = csv.writer(f)

        for idx in range(samples_nber):
            writer.writerow(
                [param_samples[prop][idx] for prop in XACRO_PROPS]
                # for solref props we only keep the sampled coeff
                + [param_samples[prop][idx][0] for prop in SOLREF_PROPS]
                + [output for output in assessment_results["outputs"][idx]]
                # inverted logic in the matlab post analysis
                + [1 - int(flag) for flag in assessment_results["flags"][idx]]
                + [thread_id])

    return True


def parallel_batch_sampler(sampling_rules: SamplingRules, tot_samples_nber: int, batch_size: int = 8):
    '''
    Samples 'sample_nber' instances in parallel on all available cores

    The total task is subdivided in batches of size 'batch_size'
    '''
    # important line to avoid issues with default xml namespace
    register_xacro_namespace()

    # prepare names and ids required later
    static_data = load_static_data_backup(RELOAD_STATIC_DATA)

    # initiate thread pool
    thread_pool = mp.Pool(mp.cpu_count())

    # number of samples per job
    batch_decomposition = np.full(tot_samples_nber // batch_size, batch_size)
    if(tot_samples_nber % batch_size != 0):
        print(
            f"WARNING: Would be more efficient if 'samples_nber' ({tot_samples_nber}) was a multiple of the batch size ({batch_size})")
        batch_decomposition = np.concatenate(
            [batch_decomposition, [tot_samples_nber % batch_size]])

    # queue all jobs and collect completion objects
    result_objs = []

    if sampling_rules.use_grid:
        # keep track of ranges for the grid case
        grid_decomposition = np.linspace(0, 1, tot_samples_nber, True)

        for idx, n_sample in enumerate(batch_decomposition):
            grid_range = grid_decomposition[idx*batch_size: (idx+1)*batch_size]
            result_objs.append(thread_pool.apply_async(
                evaluate_n_samples, args=(sampling_rules, static_data, n_sample, grid_range)))
    else:
        for n_sample in batch_decomposition:
            result_objs.append(thread_pool.apply_async(
                evaluate_n_samples, args=(sampling_rules, static_data, n_sample)))

        # wait for completion of all jobs
    _ = [res.get() for res in result_objs]

    # merge results file
    m = re.compile("results_tmp_.*.csv")
    tmp_res_files = [f for f in os.listdir(PATH_TO_RES) if (
        os.path.isfile(os.path.join(PATH_TO_RES, f)) and m.match(f) is not None)]

    # a time stamp is added to the final results file
    res_file = open(os.path.join(
        PATH_TO_RES, f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_results.csv"), "w")
    res_writer = csv.writer(res_file)

    # add header row
    res_writer.writerow(CSV_HEADER)

    # append tmp files lines
    for tmp_res_file in tmp_res_files:
        with open(os.path.join(PATH_TO_RES, tmp_res_file), 'r') as input:
            reader = csv.reader(input)
            for line in reader:
                res_writer.writerow(line)

    res_file.close()

    # clean tmp files in tmp and results folders
    remove_tmp_files(PATH_TO_TMP, f"_tmp_.*")
    remove_tmp_files(PATH_TO_RES, f"_tmp_.*")


if __name__ == "__main__":

    if __debug__:
        use_grid = False
        sampled_props = XACRO_PROPS+SOLREF_PROPS
        sampling_rules = SamplingRules(use_grid, sampled_props)

        n_samples = 1
        batch_size = 1

        parallel_batch_sampler(sampling_rules, n_samples, batch_size)
