# xlsx to csv
# convert all the xlsx tabs starting with "table"
# to individual csv files

# %%

import pandas as pd
import openpyxl
import pathlib

table_folder = pathlib.Path('csv_tables')
if not table_folder.exists():
    table_folder.mkdir()
    

src_xlsx = r"table_examples.xlsx"

wb = openpyxl.load_workbook(src_xlsx)
for sheet_name in wb.sheetnames:
    if sheet_name.startswith('table'):
        print(f'Processing {sheet_name}')
        df = pd.read_excel(src_xlsx, sheet_name=sheet_name)
        file_name = table_folder / (sheet_name + '.csv')
        df.to_csv(file_name, index=False)
    else:
        print(f'Skipping {sheet_name}')

wb.close()
print('Processing Complete')

# %%ß