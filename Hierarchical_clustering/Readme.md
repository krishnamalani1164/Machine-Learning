# Hierarchical Clustering

## Theory
Hierarchical clustering creates a tree-like hierarchy of clusters without requiring a pre-specified number of clusters. It can be agglomerative (bottom-up) or divisive (top-down), with agglomerative being more common.

## Key Formulas

### Distance Matrix
```
D = [dᵢⱼ] where dᵢⱼ = distance between points i and j
```

### Common Distance Metrics

#### Euclidean Distance
```
d(x, y) = √(Σ(i=1 to p)(xᵢ - yᵢ)²)
```

#### Manhattan Distance
```
d(x, y) = Σ(i=1 to p)|xᵢ - yᵢ|
```

#### Cosine Distance
```
d(x, y) = 1 - (x·y)/(||x||||y||)
```

## Linkage Criteria (Cluster Distance)

### Single Linkage (Minimum)
```
d(A, B) = min{d(a, b) : a ∈ A, b ∈ B}
```

### Complete Linkage (Maximum)
```
d(A, B) = max{d(a, b) : a ∈ A, b ∈ B}
```

### Average Linkage (UPGMA)
```
d(A, B) = (1/(|A||B|)) Σ(a∈A) Σ(b∈B) d(a, b)
```

### Centroid Linkage
```
d(A, B) = ||centroid(A) - centroid(B)||
```

### Ward Linkage (Minimize Within-Cluster Variance)
```
d(A, B) = √((|A||B|)/(|A|+|B|)) ||centroid(A) - centroid(B)||
```

### Weighted Average Linkage (WPGMA)
```
d(A∪B, C) = (d(A,C) + d(B,C))/2
```

## Agglomerative Algorithm Steps
1. **Initialize**: Each point as its own cluster (n clusters)
2. **Distance Matrix**: Compute pairwise distances between all clusters
3. **Merge**: Find closest pair of clusters and merge them
4. **Update**: Recalculate distances using linkage criterion
5. **Repeat**: Steps 3-4 until single cluster remains (or desired k clusters)
6. **Dendrogram**: Build tree structure showing merge hierarchy

## Lance-Williams Formula (Distance Update)
```
d(A∪B, C) = αₐd(A,C) + αᵦd(B,C) + βd(A,B) + γ|d(A,C) - d(B,C)|

Parameters for different linkages:
- Single: αₐ=αᵦ=0.5, β=0, γ=-0.5
- Complete: αₐ=αᵦ=0.5, β=0, γ=0.5
- Average: αₐ=|A|/(|A|+|B|), αᵦ=|B|/(|A|+|B|), β=0, γ=0
- Ward: αₐ=(|A|+|C|)/(|A|+|B|+|C|), αᵦ=(|B|+|C|)/(|A|+|B|+|C|), β=-|C|/(|A|+|B|+|C|), γ=0
```

## Divisive Algorithm Steps
1. **Initialize**: All points in single cluster
2. **Split**: Divide cluster using criterion (often k-means)
3. **Select**: Choose cluster to split next (largest, most heterogeneous)
4. **Repeat**: Until each point is its own cluster

## Dendrogram Analysis

### Height (Fusion Level)
```
Height = distance at which clusters were merged
```

### Cophenetic Distance
```
Cophenetic distance = height of lowest common ancestor in dendrogram
```

### Cophenetic Correlation
```
r = correlation(original_distances, cophenetic_distances)
Higher r indicates better representation
```

## Determining Number of Clusters

### Dendrogram Visual Inspection
```
Cut dendrogram at height where large jumps occur
```

### Gap Statistic
```
Gap(k) = E[log(WCSS_random)] - log(WCSS_actual)
```

### Silhouette Analysis
```
Choose k that maximizes average silhouette score
```

### Inconsistency Criterion
```
Inconsistency = (height - mean_height_below) / std_height_below
```

## Complexity Analysis
- **Time Complexity**: O(n³) for naive implementation, O(n²log n) with optimizations
- **Space Complexity**: O(n²) for distance matrix

## Linkage Properties

### Single Linkage
- **Pros**: Finds elongated clusters, handles non-convex shapes
- **Cons**: Sensitive to outliers, chain effect

### Complete Linkage
- **Pros**: Compact, spherical clusters, robust to outliers
- **Cons**: Sensitive to cluster size differences

### Average Linkage
- **Pros**: Balanced approach, good general purpose
- **Cons**: May not preserve cluster structure well

### Ward Linkage
- **Pros**: Minimizes within-cluster variance, creates compact clusters
- **Cons**: Assumes spherical clusters, sensitive to outliers

## Advantages
- No need to specify number of clusters beforehand
- Deterministic results (unlike k-means)
- Creates interpretable hierarchy
- Works with any distance metric
- Handles arbitrary cluster shapes (with appropriate linkage)

## Disadvantages
- Computationally expensive O(n³)
- Sensitive to noise and outliers
- Difficult to handle large datasets
- Once merged, cannot undo (greedy approach)
- Choice of linkage criterion affects results significantly

## Use Cases
- **Phylogenetic Analysis**: Evolutionary relationships
- **Social Network Analysis**: Community detection
- **Gene Expression**: Grouping genes by expression patterns
- **Image Segmentation**: Pixel grouping
- **Market Segmentation**: Customer hierarchy
- **Document Clustering**: Topic hierarchies
- **Taxonomy Creation**: Organizing data into hierarchies
