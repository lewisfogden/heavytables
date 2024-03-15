# %%
# Convert CMI tables to long format (from tab separated csv)

import pandas as pd

src = 'cmi_TMNL16_tab_fake.xlsx'

df_src = pd.read_excel(src, comment='#')
df_src
# %%

df = df_src.set_index('age').stack().reset_index()
df.rename(columns={'age':'age|int',
                   'level_1':'dur|int',
                   0:'q_x|float'}, inplace=True)

# make sure all dur|int are integers

df
# %%
df.to_csv('fake_tmnl16.csv', index=False)