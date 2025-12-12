import re
import pandas as pd

df = pd.read_excel("mbpp_js.xlsx")
pattern = re.compile(r"console\.log")
for i in range(len(df)):
    df['js_test'][i] = pattern.sub("console.assert",df['js_test'][i])

df.to_excel("mbpp_js_processed.xlsx")