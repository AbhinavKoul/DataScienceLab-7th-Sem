import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv("2 retail_dataset.csv")

itemset = set(df.values.flatten())

#for each row, check if that element is present or not
encoded_values = []

for idx,row in df.iterrows():
    rowset = set(row)
    labels = {}
    
    uncommon = list(itemset - rowset)
    common = list(itemset.intersection(rowset))
    
    for uc in uncommon:
        labels[uc] = 0
    
    for com in common:
        labels[com] = 1
        
    encoded_values.append(labels)

tf_df = pd.DataFrame(encoded_values)

#apriori
freq_items = apriori(tf_df, min_support=0.2, use_colnames=True)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

# GRAPHS
# Rules vs Confidence
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
# Support vs Lift
plt.scatter(rules['support'], rules['lift'])
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift')
plt.show()
# Lift vs Confidence
fit = np.polyfit(rules['lift'], rules['confidence'], 1)
fit_fn = np.poly1d(fit)
plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], fit_fn(rules['lift']))