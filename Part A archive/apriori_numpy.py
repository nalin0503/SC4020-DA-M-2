from load_data import load_data
import numpy as np
from queue import Queue

# Load the data
baskets = [
    [1,2, 3, 4],
    [1, 2, 3],
    [1, 2, 3, 4],
    [1, 2, 3, 4,5],
    [1, 2, 3, 4, 5, 6]
]

# for city in 'B', 'C', 'D':
#     raw = load_data(f'POIdata_city{city}.csv')
#     groupd = raw.groupby(['x', 'y'], as_index=False).agg({'category': list})
#     baskets.extend(groupd['category'].tolist())

# C# Identify all unique items
unique_items = sorted(set(item for basket in baskets for item in basket))

# Create a binary matrix
binary_matrix = np.zeros((len(baskets), len(unique_items)), dtype=int)

# Fill the binary matrix
item_index = {item: idx for idx, item in enumerate(unique_items)}
for i, basket in enumerate(baskets):
    for item in basket:
        binary_matrix[i, item_index[item]] = 1

    
# Apriori Settings
minsup = 3 # Item must at least appear 10 times
minconf = 70

def generate_freq_itemsets(baskets, minsup):
    # we will do this in a dfs manner
    
    raw_item_indices = np.arange(1, len(unique_items)+1)
    print(raw_item_indices)
    
    # Count the number of times each item appears
    item_counts = binary_matrix.sum(axis=0)
    
    # Filter out items that do not meet the minsup threshold
    freq_items = item_counts >= minsup
    
    # Initialize the queue
    q = Queue()
    
    # Add the frequent items to the queue
    for idx, item in enumerate(freq_items):
        if item:
            # Convert the index to a numpy array of zeros with a 1 at the index
            new_item = np.zeros(len(unique_items), dtype=int)
            new_item[idx] = 1
            q.put(new_item)
    
    freq_itemsets = []
    
    # Generate candidate sets
    while not q.empty():
        
        itemset_mask = q.get()
        freq_itemsets.append(itemset_mask)
        
        # We are going to find the first item in lexigraphical order that is not in the itemset. Meaning, we will find the first 0 in the itemset after 
        # the contiguous string of 0s and 1s
        ptr = 0
        
        # Jump over the zeros
        for idx in range(len(itemset_mask)):
            if itemset_mask[idx] == 1:
                ptr = idx
                break
            

        # Jump over the ones
        while itemset_mask[ptr] == 1:
            ptr += 1
            if ptr == len(itemset_mask):
                break
        
        # If we have reached the end of the itemset, we are done
        if ptr == len(itemset_mask):
            continue

        for idx in range(ptr, len(itemset_mask)):
            
            if itemset_mask[idx]:
                continue
        
            # Create a new itemset by copying the current itemset and setting the next item to 1
            new_itemset_mask = itemset_mask.copy()
            new_itemset_mask[idx] = 1
            
            # Count the frequency of the new itemset mask in the binary matrix
            freq = np.sum(np.all((binary_matrix & new_itemset_mask) == new_itemset_mask, axis=1))
            
            # If the frequency is greater than the minsup, add it to the queue
            if freq >= minsup:
                q.put(new_itemset_mask)
                print(raw_item_indices[new_itemset_mask==1])
    
    return freq_itemsets
            

freq = generate_freq_itemsets(baskets, minsup)

print(freq)
