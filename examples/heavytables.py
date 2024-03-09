# heavytables.py
# %%
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})

class BandIndex:
    def __init__(self, upper_bound_array, band_label_array):
        """Inputs must be sorted"""
        self.upper_bound_array = np.array(upper_bound_array)
        self.band_label_array = np.array(band_label_array)

    def get(self, numpy_array):
        """get the labels for vector"""
        indices = np.searchsorted(self.upper_bound_array, numpy_array)
        return self.band_label_array[indices]
    
    def get_max(self, numpy_array):
        """get the upper band limit"""
        indices = np.searchsorted(self.upper_bound_array, numpy_array)
        return self.upper_bound_array[indices]
    
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
    band_table = BandIndex.from_excel_sheet(src_file, sheet_name)

    table_band_test = np.array(
        [0, 5000, 10251, 256640]
    )
    band_table.get(table_band_test)

class NumTable:
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
    tab_int_int = NumTable.read_excel(src_file, sheet_name="table_int_int")

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