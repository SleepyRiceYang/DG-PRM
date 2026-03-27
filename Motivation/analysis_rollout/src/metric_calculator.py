import pandas as pd
import numpy as np

class MetricCalculator:
    def __init__(self, df_turns):
        self.df = df_turns.sort_values(by=['User', 'TrajID', 'Turn'])

    def compute_basic_metrics(self):
        # 1. Dissonance
        self.df['Dissonance'] = self.df['Hs'] - self.df['Ha']

        # 2. Delta (Trend components)
        self.df['Delta_Hs'] = self.df.groupby(['User', 'TrajID'])['Hs'].diff().fillna(0)
        self.df['Delta_Ha'] = self.df.groupby(['User', 'TrajID'])['Ha'].diff().fillna(0)

        # 3. Trend Momentum
        self.df['Trend'] = self.df['Delta_Hs'] - self.df['Delta_Ha']
        
        return self.df
