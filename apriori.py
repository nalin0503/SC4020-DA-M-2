from itertools import combinations
from collections import defaultdict
from load_data import load_data

def apriori(baskets, minsup, minconf):
    # Step 1: Generate Frequent Itemsets
    def get_frequent_itemsets(baskets, minsup):
        itemsets = defaultdict(int)
        for basket in baskets:
            for item in basket:
                itemsets[frozenset([item])] += 1

        # Filter itemsets by minsup
        itemsets = {k: v for k, v in itemsets.items() if v >= minsup}
        frequent_itemsets = [set(itemsets.keys())]
        print(len(frequent_itemsets[0]))

        k = 2
        while frequent_itemsets[-1]:
            candidate_itemsets = defaultdict(int)
            for basket in baskets:
                for itemset in combinations(set(basket), k):
                    itemset = frozenset(itemset)
                    if all(frozenset(subset) in frequent_itemsets[-1] for subset in combinations(itemset, k - 1)):
                        candidate_itemsets[itemset] += 1

            candidate_itemsets = {k: v for k, v in candidate_itemsets.items() if v >= minsup}
            frequent_itemsets.append(set(candidate_itemsets.keys()))
            print(f"Generated {len(frequent_itemsets[-1])} frequent itemsets of length {k}")
            k += 1

        return [itemset for level in frequent_itemsets for itemset in level if itemset]

    # Step 2: Generate Association Rules
    def generate_rules(frequent_itemsets, minconf):
        rules = []
        for itemset in frequent_itemsets:
            for consequent_len in range(1, len(itemset)):
                for consequent in combinations(itemset, consequent_len):
                    consequent = frozenset(consequent)
                    antecedent = itemset - consequent

                    if antecedent:
                        support_antecedent = sum(1 for basket in baskets if antecedent.issubset(basket))
                        support_itemset = sum(1 for basket in baskets if itemset.issubset(basket))
                        confidence = (support_itemset / support_antecedent) * 100

                        if confidence >= minconf:
                            rules.append((antecedent, consequent, confidence))

        return rules

    frequent_itemsets = get_frequent_itemsets(baskets, minsup)
    rules = generate_rules(frequent_itemsets, minconf)
    return rules

baskets = []

for city in 'B', 'C', 'D':
    raw = load_data(f'POIdata_city{city}.csv')
    groupd = raw.groupby(['x', 'y'], as_index=False).agg({'category': list})
    baskets.extend(groupd['category'].tolist())

minsup = 2000
minconf = 80
association_rules = apriori(baskets, minsup, minconf)
for rule in association_rules:
    print(f"Rule: {rule[0]} -> {rule[1]}, Confidence: {rule[2]:.2f}%")
