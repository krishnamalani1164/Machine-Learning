# ECLAT Algorithm for Association Rule Learning

## Overview
ECLAT (Equivalence Class Clustering and bottom-up Lattice Traversal) is an efficient algorithm for mining frequent itemsets using vertical data representation.

## Key Concepts

### Vertical Data Format
Transform horizontal transactions into vertical item lists:
```
Horizontal: T1:{A,B}, T2:{A,C}, T3:{B,C}
Vertical: A:{T1,T2}, B:{T1,T3}, C:{T2,T3}
```

### Support Calculation
```
Support(itemset) = |TID-list intersection| / |total transactions|
```

### TID-list Intersection
For itemset {A,B}: TID(A) ∩ TID(B)
```
A:{1,2,4} ∩ B:{1,3,4} = {1,4}
Support({A,B}) = 2/4 = 50%
```

## Algorithm Steps
1. **Convert to vertical format** - Create TID-lists for each item
2. **Generate candidates** by intersecting TID-lists
3. **Calculate support** from intersection size
4. **Prune infrequent** itemsets below minimum support
5. **Recursively process** frequent itemsets using DFS
6. **Generate association rules** from frequent itemsets

## ECLAT Property
If TID-list intersection is empty or below threshold, prune the branch.

## Example
```
Transactions:
T1: {A,B,D}
T2: {A,C,E} 
T3: {B,C,D}
T4: {A,B,C}

Vertical Format:
A: {1,2,4}
B: {1,3,4}  
C: {2,3,4}
D: {1,3}
E: {2}

Min Support = 50% (2/4)

2-itemsets:
{A,B}: {1,4} → Support = 50% ✓
{A,C}: {2,4} → Support = 50% ✓
{B,C}: {3,4} → Support = 50% ✓

3-itemsets:
{A,B,C}: {4} → Support = 25% ✗
```

## Advantages
- **Memory efficient** - No candidate generation
- **Faster intersections** - Uses TID-lists directly  
- **Depth-first search** - Better for dense datasets
- **Parallel processing** - Easy to distribute

## vs Apriori
| Feature | ECLAT | Apriori |
|---------|-------|---------|
| Data Format | Vertical | Horizontal |
| Search Strategy | DFS | BFS |
| Memory Usage | Lower | Higher |
| Best For | Dense data | Sparse data |

## Applications
- Market basket analysis
- Web clickstream mining
- Bioinformatics patterns
- Social network analysis
