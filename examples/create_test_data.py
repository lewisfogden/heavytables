# %%

import numpy as np
import pandas as pd

sample_sizes = [1, 100, 10_000, 100_000]

rng = np.random.default_rng(seed=42)

for policies in sample_sizes:
    data = dict(
        mp_num = np.arange(policies),
        init_age = rng.integers(low=20, high=75, size=policies),
        initial_pols_if = np.ones(policies),
        sum_assured = rng.uniform(10_000, 250_000, policies).round(0)
    )
    df = pd.DataFrame(data)
    df.to_csv(f'data/data_{policies}.csv', index=False)

# %%
