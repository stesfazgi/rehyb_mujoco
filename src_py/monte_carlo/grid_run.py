'''
Script allowing grid sampling of models;
To be run on the server with the -O flag
'''

from tqdm import tqdm

from monte_carlo.sample_models import parallel_batch_sampler, SamplingRules, SAMPLING_KEYS

if __name__ == "__main__":
    grid_size = 100
    batch_size = 4

    for name in tqdm(SAMPLING_KEYS):
        sampling_rules = SamplingRules(True, [name])

        parallel_batch_sampler(sampling_rules, grid_size, batch_size)
