import pandas as pd
import numpy as np

dev = pd.read_csv('dev.csv')
dev_winners = pd.read_csv('dev_winners.csv')
dev['is_winner'] = dev['candidate_id'].isin(dev_winners['candidate_id']).astype(int)

# parse numeric
def clean_score(s):
    try:
        return float(str(s).split('/')[0].strip())
    except:
        return np.nan

dev['tech'] = dev['technical_assessment'].apply(clean_score)
dev['apt'] = dev['aptitude_score'].apply(clean_score)
dev['rating'] = dev['last_rating'].apply(clean_score)

print("Correlation with is_winner:")
corrs = dev.select_dtypes(include=[np.number]).corr()['is_winner'].sort_values(ascending=False)
print(corrs.head(20))
print(corrs.tail(5))
