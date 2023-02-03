'''
Script allowing concurrent sampling of models;
To be run on the server with the -O flag
'''

from tqdm import tqdm

from monte_carlo.sample_models import parallel_batch_sampler, SamplingRules, SAMPLING_KEYS

if __name__ == "__main__":
    batch_size = 2**4
    n_subsampling = 2**4
    subsampling_size = 2**12

    sampling_rules = SamplingRules(False, list(SAMPLING_KEYS))

    for _ in tqdm(range(n_subsampling)):
        parallel_batch_sampler(sampling_rules, subsampling_size, batch_size)
