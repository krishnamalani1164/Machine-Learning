# K-Means Clustering

## Theory
K-Means is an unsupervised clustering algorithm that partitions data into k clusters by minimizing within-cluster sum of squares. It iteratively assigns points to nearest centroids and updates centroids until convergence.

## Key Formulas

### Objective Function (Within-Cluster Sum of Squares)
```
WCSS = Σ(i=1 to k) Σ(x∈Cᵢ) ||x - μᵢ||²

Where:
- k = number of clusters
- Cᵢ = cluster i
- μᵢ = centroid of cluster i
- x = data point
```

### Centroid Update
```
μᵢ = (1/|Cᵢ|) Σ(x∈Cᵢ) x

Where |Cᵢ| = number of points in cluster i
```

### Distance Metric (Euclidean)
```
d(x, μᵢ) = √(Σ(j=1 to p)(xⱼ - μᵢⱼ)²)

Where p = number of features
```

### Cluster Assignment
```
C(x) = argmin(i=1 to k) ||x - μᵢ||²
```

### Inertia (Total WCSS)
```
Inertia = Σ(i=1 to n) min(μ∈centroids) ||xᵢ - μ||²
```

## Algorithm Steps
1. **Initialize**: Choose k and randomly place k centroids
2. **Assignment Step**: Assign each point to nearest centroid
   ```
   For each point x: cluster(x) = argmin(d(x, μᵢ))
   ```
3. **Update Step**: Recalculate centroids as cluster means
   ```
   For each cluster i: μᵢ = mean(all points in cluster i)
   ```
4. **Repeat**: Steps 2-3 until convergence (centroids don't move)
5. **Convergence**: When Δμᵢ < tolerance or max_iterations reached

## Initialization Methods

### Random Initialization
```
μᵢ = random point from data range
```

### K-Means++ (Smart Initialization)
```
1. Choose first centroid randomly
2. For remaining centroids:
   - Calculate D(x) = distance to nearest existing centroid
   - Choose next centroid with probability ∝ D(x)²
```

### Forgy Method
```
Randomly select k data points as initial centroids
```

## Convergence Criteria

### Centroid Movement
```
Convergence when: max(||μᵢ_new - μᵢ_old||) < tolerance
```

### Cost Function Change
```
Convergence when: |WCSS_new - WCSS_old| < tolerance
```

### Maximum Iterations
```
Stop after max_iter iterations regardless of convergence
```

## Choosing Optimal k

### Elbow Method
```
Plot WCSS vs k, find "elbow" where rate of decrease slows
```

### Silhouette Score
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest different cluster
- Range: [-1, 1], higher is better
```

### Gap Statistic
```
Gap(k) = E[log(WCSS_random)] - log(WCSS_actual)
Choose k where Gap(k) is maximized
```

## Key Parameters
- **k**: Number of clusters (must be specified)
- **init**: Initialization method ('k-means++', 'random')
- **n_init**: Number of random initializations (default: 10)
- **max_iter**: Maximum iterations (default: 300)
- **tol**: Tolerance for convergence (default: 1e-4)
- **algorithm**: 'lloyd', 'elkan', 'auto'

## Variants

### Mini-Batch K-Means
```
Use random subsets (mini-batches) for faster computation
Trade-off: Speed vs. accuracy
```

### K-Means++
```
Smart initialization to improve convergence and quality
```

## Assumptions
- **Spherical clusters**: Assumes clusters are roughly circular
- **Similar sizes**: Works best when clusters have similar sizes
- **Similar density**: Assumes uniform density within clusters
- **Well-separated**: Clear separation between clusters

## Advantages
- Simple and fast algorithm
- Guaranteed convergence
- Works well with spherical clusters
- Computationally efficient O(tkn)
- Good for large datasets

## Disadvantages
- Must specify k in advance
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers
- Struggles with clusters of different sizes/densities

## Distance Metrics (Alternatives)

### Manhattan Distance
```
d(x, y) = Σ|xᵢ - yᵢ|
```

### Cosine Distance
```
d(x, y) = 1 - (x·y)/(||x||||y||)
```

## Use Cases
- **Customer Segmentation**: Group customers by behavior
- **Image Segmentation**: Pixel grouping in computer vision
- **Market Research**: Consumer preference groups
- **Data Compression**: Vector quantization
- **Anomaly Detection**: Identify outliers from clusters
- **Preprocessing**: Feature engineering for supervised learning
