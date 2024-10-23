# Import necessary libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Create a simple dataset (Transaction Data)
# In a real-world scenario, this would be your transaction data
dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs', 'Bread'],
    ['Eggs'],
    ['Bread']
]

# Step 2: Convert the dataset into a pandas DataFrame suitable for apriori
# Create a set of unique items in the dataset
items = sorted(set(item for transaction in dataset for item in transaction))

# Create an empty DataFrame where rows are transactions and columns are items
df = pd.DataFrame([[1 if item in transaction else 0 for item in items] for transaction in dataset], columns=items)

print("Transaction Data (One-hot Encoded):")
print(df)

# Step 3: Apply Apriori to find frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate association rules from frequent itemsets
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("\nAssociation Rules:")
print(rules)

# Step 5: Filter and display rules with high confidence
print("\nRules with Confidence > 0.8:")
high_conf_rules = rules[rules['confidence'] > 0.8]
print(high_conf_rules)
