import pandas as pd
import numpy as np
import os,config

general = pd.read_csv(config.path_to_info + "candidates_v2.csv")
specified = pd.read_csv("candidates_v1.csv")

result = general[general['class'].isin([0])]
#print(result)
result = pd.concat([specified,result],axis=0)
#print(result)
result = result.reset_index(drop=True)
result.to_csv("final_candidates.csv",index=False,header=True)