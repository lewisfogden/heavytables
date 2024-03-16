# %%

import heavylight # pip install heavylight
import pandas as pd
import numpy as np
import seaborn as sns

import heavytables

# load in a few tables
tab1 = heavytables.Table.read_csv(r'csv_tables/table_band.csv')
tab2 = heavytables.Table.read_csv(r'csv_tables/table_str_int_band.csv')
mort = heavytables.Table.read_csv(r'csv_tables/fake_tmnl16.csv')

# check the tables work before using them in the model.
print(tab1[1234])    # single value
print(tab1[np.array([1234])])    # single array value
print(tab1[np.array([1234, 1234]),])  # multiple values

print(tab2["ABC", 2023, 9999])
print(tab2["ABC", 2023, 10001])
print(mort[50, 5])

def mort_calc(age, dur):
    # test of mortality tables, uses same formula as spreadsheet
    # EXP($A4*0.1-13-(5-B$3)/5)
    capped_dur = np.clip(dur,0, 5)
    return np.exp(0.1 * age - 13 - (5 - capped_dur)/5)

# %%

class TableTest(heavylight.Model):
    def t(self, t):
        return t
    
    def unit_fund(self, t):
        if t == 0:
            return self.data['init_fund']
        else:
            return self.unit_fund(t - 1) * (1 + 0.04/12) - self.unit_charge(t - 1) - self.monthly_mort_charge(t - 1)
    
    def unit_charge(self, t):
        return self.basis['cost_table'][self.unit_fund(t)]
    
    def annual_expense(self, t):
        return self.basis['exp_tab'][self.data['product'], self.capped_year(t), self.unit_fund(t)]
    
    def capped_year(self, t):
        return np.clip(self.year(t), 2023, 2025)
    
    def year(self, t):
        if t == 0:
            return self.data['proj_year']
        elif t % 12 == 0:
            return self.year(t - 1) + 1
        else:
            return self.year(t - 1)
    
    def dur_if(self, t):
        #duration in force for policy (months) at time t
        return self.data['init_dur_m'] + t

    def age(self, t):
        if t == 0:
            return self.data['init_age']
        elif t % 12 == 0:
            return self.age(t - 1) + 1
        else:
            return self.age(t - 1)

    def annual_mort(self, t):
        capped_dur = np.clip(self.dur_if(t), 0, 5)
        return self.basis['mort'][self.age(t), capped_dur]

    def monthly_mort(self, t):
        return (1 - (1 - self.annual_mort(t)) ** (1/12))
    
    def sum_at_risk(self, t):
        """insured amount - provides guarantee on death"""
        return np.maximum(self.data['sum_assured'] - self.unit_fund(t), 0)
    
    def monthly_mort_charge(self, t):
        """change for guaranteed minimum death benefit
        """
        return self.sum_at_risk(t) * self.monthly_mort(t)

# simulate some data - store in a dictionary to pass into the model
rng = np.random.default_rng(seed=42)
policies = 100000

data = dict(
    init_fund = rng.uniform(low=1000, high=250000, size=policies),
    init_dur_m = rng.integers(low=0, high = 10*12, size=policies),
    init_age = rng.integers(low = 18, high = 65, size=policies),
    proj_year = np.repeat([2023], policies),
    product = rng.choice(['ABC', 'DEFG', 'HIJKL'], size=policies),
    sum_assured = rng.uniform(low=200_000, high=500_000, size=policies),
)

# store basis items in a dictionary

basis = dict(
    cost_table = tab1,
    exp_tab = tab2,
    mort = mort,
)

# run the projection

proj_result = TableTest(do_run=True, proj_len=240, data=data, basis=basis)
proj_result.ToDataFrame() # view dataframe of result (output isn't nice yet!)


# %%
# Do some plots of results (quick checks)
xs = proj_result.unit_fund(240)
ys = proj_result.unit_charge(240)

sns.scatterplot(x=xs, y=ys, marker='+');
# %%
sns.scatterplot(x=proj_result.unit_fund(240),
                y=proj_result.annual_expense(240),
                hue=data['product'],
                marker='.');
# %%
sns.scatterplot(x=proj_result.unit_fund(240),
                y=proj_result.monthly_mort_charge(240),
                hue=data['product'],
                marker='.');
# %%
