# heavytables.py
# %%
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

class SelfLookup:
    def __int__(self, *args, **kwargs):
        pass

    def get(self, values):
        return values

class BandLookup:
    def __init__(self, upper_bounds, labels):
        """Inputs must be sorted"""
        self.upper_bounds = np.array(upper_bounds)
        self.labels = np.array(labels)

    def get(self, numpy_array):
        """get the labels for vector"""
        indices = np.searchsorted(self.upper_bounds, numpy_array)
        return self.labels[indices]
    
    def get_max(self, numpy_array):
        """get the upper band limit"""
        indices = np.searchsorted(self.upper_bounds, numpy_array)
        return self.upper_bounds[indices]
    
    @classmethod
    def from_dataframe(cls, df:pd.DataFrame, band_column_name, integer_column_name):
        df_copy = df.sort_values(band_column_name)
        band_arr = df_copy[band_column_name].values
        band_labels = df_copy[integer_column_name].values
        return cls(band_arr, band_labels)
    
    @classmethod
    def from_excel_sheet(cls, workbook_path, sheet_name):
        """Extract from an excel sheet, there must be two columns, the band column and the integer column"""
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        assert len(df.columns) == 2, "must have 2 columns"
        band_col_name, int_col_name = df.columns[0], df.columns[1]
        return cls.from_dataframe(df, band_col_name, int_col_name)

    
if __name__ == "__main__":
    src_file = "table_examples.xlsx"
    sheet_name = "table_band"
    band_table = BandLookup.from_excel_sheet(src_file, sheet_name)

    table_band_test = np.array(
        [0, 5000, 10251, 256640]
    )
    band_table.get(table_band_test)

class IntKeyTable:
    """Vectorised Table Allowing for multiple keys
    
    Keys are all integers
    Use BandIndex for non integer keys (covert to 0..1...N)
    """

    def __init__(self, df: pd.DataFrame):
        self.key_cols = list(df.columns[:-1])
        self.value_col = df.columns[-1]
        self.df = df.sort_values(by=list(reversed(self.key_cols)))
        self.bases = [min(self.df[col]) for col in self.key_cols]
        self.ranges = [max(self.df[col]) - min(self.df[col]) + 1 for col in self.key_cols]

        # set up index scalars
        total_scalar = 1
        self.scalars = []
        for col in self.ranges:
            self.scalars.append(total_scalar)
            total_scalar *= col

        self.values = self.df[self.value_col].values

        expected_rows = np.prod(self.ranges)
        assert expected_rows == len(self.values)

    def get_index(self, *keys):
        index = 0
        for key, base, scalar in zip(keys, self.bases, self.scalars):
            index += (key - base) * scalar
        return index
    
    def get_value(self, *keys):
        indices = self.get_index(*keys)
        return self.values[indices]
    
    def __getitem__(self, keys):
        return self.get_value(*keys)
    
    @classmethod
    def read_excel(cls, spreadsheet_path, sheet_name):
        df = pd.read_excel(spreadsheet_path, sheet_name=sheet_name)
        return cls(df)
    
if __name__ == "__main__":
    tab_int_int = IntKeyTable.read_excel(src_file, sheet_name="table_int_int")

    test_ages = np.array([18, 22, 24, 30, 18])
    test_years = np.array([2023, 2024, 2023, 2043, 2043])
    test_expected = test_ages * test_years / 100_000
    print(tab_int_int[test_ages, test_years])
    print(tab_int_int[test_ages, test_years] - test_expected)

    min_test_age = 18
    max_test_age = 30
    min_test_year = 2023
    max_test_year = 2043

    # make loads of test cases!
    sims = 1_000_000
    rng = np.random.default_rng(42)
    ages = rng.integers(low=min_test_age, high=max_test_age + 1, size=sims)
    years = rng.integers(low=min_test_year, high=max_test_year + 1, size=sims)
    rate_output = tab_int_int[ages, years]
    rate_expected = ages * years / 100_000
    print("All Close? ", np.allclose(rate_output, rate_expected))

# %%
class Table:
    """A generic table class, allows for any keys
    """
    col_types = "int", "int_bound", "str", "band", "float"

    def __init__(self, df:pd.DataFrame):
        # scan columns
        
        # convert non-integer keys to integers
        key_cols = list(df.columns[:-1])
        df = df.sort_values(key_cols[::-1]) # sort by reverse order

        # col_types = [col.split("|")[1] for col in key_cols]

        self.mappers = []
        for col in key_cols:
            col_type = col.split("|")[1] # "int", "str" etc
            if col_type == "int":
                self.mappers.append(SelfLookup()) # just so we have .get (a bit inefficient?)
            elif col_type in ["str", "band"]:
                df_col = pd.DataFrame(df[col].unique(), columns=["band_name"]).reset_index().sort_values("band_name")
                band_mapper = BandLookup.from_dataframe(df_col, "band_name", "index")
                self.mappers.append(band_mapper)
            else:
                raise NotImplementedError(f"{col_types} not implemented on {col}")
        
        # create an intkeytable
        # prepare the right keys

        self._int_key_table = IntKeyTable


        # assign mappers for non-integer types

        # for integer types - skip or use a neutral mapper?

# may a custom designed lookup
    
# %%
df = pd.read_excel(src_file, sheet_name="table_str_int_band")
key_cols = list(df.columns[:-1])

df = df.sort_values(key_cols[::-1]) # put the columns in order
df

# get the column types
col_types = [col.split("|")[1] for col in key_cols]
col_types

# generate mappers for non-integer types
# str
#product|str
df_product = pd.DataFrame(df[key_cols[0]].unique(), columns=["band_name"]).reset_index().sort_values("band_name")
product_band = BandLookup.from_dataframe(df_product, "band_name", "index")
test_products = np.array(["ABC", "DEFG", "IJKLM", "IJKLM", "ABC"])
print(product_band.get(test_products))

df_fund_to = pd.DataFrame(df[key_cols[2]].unique(), columns=["band_name"]).reset_index().sort_values("band_name")
fund_to_band = BandLookup.from_dataframe(df_fund_to, "band_name", "index")
test_bands = np.linspace(0, 1_300_000, 1000)
print(fund_to_band.get(test_bands))

df_int_keys = df.copy()
df_int_keys["product|int"] = product_band.get(df_int_keys["product|str"])
df_int_keys["fund_to|int"] = fund_to_band.get(df_int_keys["fund_to|band"])
df_int_keys = df_int_keys[["product|int", "year|int", "fund_to|int", "value|float"]]
df_int_keys
# %%
table_df = IntKeyTable(df_int_keys)

# now to test - sample a lot

sample_df = df.sample(1_000_000, replace=True, random_state=42)

# now we need to look up using translated keys

sample_keys = product_band.get(sample_df["product|str"]), sample_df["year|int"].values, fund_to_band.get(sample_df["fund_to|band"])

sample_df["result"] = table_df[sample_keys]
sample_df["diff"] = sample_df["result"] - sample_df["value|float"]
print(np.allclose(sample_df["result"], sample_df["value|float"]))
print(sample_df["diff"].describe())


# %%
