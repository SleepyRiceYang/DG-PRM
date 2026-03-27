import pandas as pd

class TrendAnalyzer:
    def __init__(self, df_turns_metrics):
        self.df = df_turns_metrics

    def analyze_trends(self):
        """2.1 按 Outcome 聚合四维指标"""
        # 计算均值和置信区间所需的数据 (Seaborn 会自动处理 CI，这里只需返回长格式数据)
        return self.df[['Turn', 'Outcome', 'Hs', 'Ha', 'Dissonance', 'Trend']]
