# Use BACI's HS12 revision rather than the newest HS revision

BACI publishes the same trade data under several HS revisions. We use HS12 (2012 onwards) rather than HS17 or HS22 because the model only learns Scenario Effects from Trade Agreements that came into force during the training years. A longer history gives far more such changes, and it also covers the EU–Japan Event Backtest (2019) with a proper pre-period. The cost is that Product codes are 2012 vintage: a few HS6 codes Advisors know from current tariff schedules will differ.
