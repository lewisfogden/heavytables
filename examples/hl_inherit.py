# %%
from heavylight import Model

# %%

class Core(Model):
    def t(self, t):
        return t
    
    def proj_month(self, t):
        """projection month: 1=Jan, 12=Dec
        
        Assumes projection starts in January (could override using "init_proj_month")
        """
        if t == 0:
            return self.data['init_proj_month']
        elif self.proj_month(t - 1) == 12:
            return 1
        else:
            return self.proj_month(t - 1) + 1
    
    def proj_year(self, t):
        if t == 0:
            return self.data["init_proj_year"]
        elif self.proj_month(t - 1) == 12:
            return self.proj_year(t - 1) + 1
        else:
            return self.proj_year(t - 1)
        
if __name__ == '__main__':
    core_data = dict(
        init_proj_month = 3,
        init_proj_year = 2024,
    )
    proj = Core(do_run=True, proj_len=48, data=core_data)
    print(proj.ToDataFrame())


# %%
class Life1(Model):
    #note - this isn't vectorised, need to use np.where
    def age_years(self, t):
        if t == 0:
            return self.data['init_age_years']
        elif self.age_months(t - 1) == 12:
            return self.age_years(t - 1) + 1
        else:
            return self.age_years(t - 1)
    
    def age_months(self, t):
        if t == 0:
            return self.data['init_age_months']
        elif self.age_months(t - 1) == 12:
            return 1
        else:
            return self.age_months(t - 1) + 1
        
    def qx_annual(self, t):
        return 0.001 * self.age_years(t)
    
    def qx_monthly(self, t):
        return self.qx_annual(t) / 12 # lazy, example

if __name__ == '__main__':
    l1_data = dict(
        init_age_years = 38,
        init_age_months=8,
    )
    proj_l1 = Life1(proj_len=24, do_run=True, data=l1_data)
    print(proj_l1.ToDataFrame())

# %%
# combined model (don't actually need to inherit Model directly)
class Term(Core, Life1, Model):
    def claim(self, t):
        return self.qx_monthly(t) * self.data['sum_assured']

if __name__ == '__main__':
    term_data = l1_data | core_data | dict(
        sum_assured = 10_000,
    )
    proj_term = Term(do_run=True, proj_len=36, data=term_data)
    print(proj_term.ToDataFrame())
# %%
