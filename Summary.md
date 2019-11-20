# Summary of ECAC's theoretical material

## Clustering

### Difference between values of common attribute types
* __Quantitative attributes__
    * _d(a,b)_ = |a - b|
* __Qualitative attributes__
    * ordinal
        * _d(a,b)_ = (|rA - rB|) / (n - 1)
        * with rX being the ranking position of the X value
    * nominal
        * _d(a,b)_ = a == b ? 0 : 1

### Distance measures for objects with quantitative attributes
* Manhattan distance (r=1) -> 'rigid' line from X to Y 
* Euclidean distance (r=2) -> straight line from X to Y
* Notice that objects with mixed attribute types can usually be transformed to only quantitative attributes

### Difference between sequences
* __Hamming distance__
    * The number of positions at which the corresponding characters in the two strings are different
        * _e.g._ d("james", "jimmy") = 3
* __Levenshtein or edit distance__
    * Minimum number of operations necessary to transform one
sequence into another (insertion, removal or substitution)
        * _e.g._ d("jhonny", "jonston") = 5
* __Bag-of-words vector similarity__
    * Long texts converted to a quantitative vector, where the difference of the texts is the distance between their bag-of-word vectors
    * Each position is associated with one of the words found and its value is the number of times this word appeared in the text
        * _e.g._
        * textA = "I will go to the party. But first, I will have to work"
        * textB = "they have to go to the work by bus"
        * d = 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 = 10

| | I | will | go | to | the | party | but | first | have | work | they | by | bus |
|:-:|-:| -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| A | 2 | 2 |1 |2 |1 | 1 |1 | 1 | 1 |1 |0 |0 |0 |
| B | 0 | 0 |1 |2 |1 | 0 |0 | 0 | 1 |1 |1 |1 |1 |

* __Dynamic Time Warping__
    * Similar to the edit distance
    * Returns the value of the best match between the two time series (considering variations and shifts in time and speed)

### Distance between images
* __Features representation__
    *  Features associated with the application are extract from the image
        * _e.g._ the distance between the eyes
    * Image is represented by a vector of real numbers, each element corresponding to one particular feature
* __Matrix / vector of pixels__
    * Each pixel is an integer
    * Size of the area is associated with the granularity required for the image
    * _e.g_ in image below
        * Distance value is then calculated to process similar to the bag of words
![Vector of pixels example](https://i.imgur.com/oVFSH9v.png)

### Clustering techniques
* __Partitional__
    * Each object in the cluster is closer to every other object in the cluster than to any object outside the cluster 
* __Prototype-based__
    * Each object in the cluster is closer to a prototype representing the cluster than to a prototype representing any other cluster

Main type of clusters:
* Graph-based
    * Represents the data set by a graph structure
        * Connecting objects that belong to the same cluster with an edge (objects are nodes)
* Density-based
    * A cluster is a region with high density of objects surrounded by a region of low density


#### __K-means__
* Most popular
* Creates clusters that can be both partitional or prototype-based
    * __Centroid (prototype-based)__: A prototype or profile of all the objects in a cluster, for example the average of all the objects in the cluster
    * _e.g._ (another example in slides)
![k-means centroid example](https://i.imgur.com/sc1dmm2.png)
    * __Medoid (partitional)__ : The instance in a cluster with the shortest average distance to all other instances in the cluster
* Pros
    * Computationally efficient
    * Often finds good results
* Cons
    * Can get stuck in local optima
    * Multiple runs with re-initializations may be needed
    * Necessary to define _K_ in the beginning
    * Does not work well with noisy data and outliers
    * Can find only convex shaped clusters
* Find optimal number of clusters -> Within-groups sum of squares technique -> Elbow curve -> optimal _K_ is a the elbow (see slides for more info)

#### __DBSCAN__
* Creates clusters that are partitional and density-based
* Good to find non-convex (one or more of the interior angles of the shape is > 180º) clusters
* Automatically defines the number of clusters
* Objects forming a dense region belong to the same cluster
* Objects not belonging to dense regions are considered to be noise
* __Core instance__: Directly reaches a minimum number (_delta_) of other instances
* Key concept: __reachability__
    * If _p_ is a core instance then it forms a cluster with all instances it reaches both directly and indirectly
        * Reaching _q_ directly means that the distance between _p_ and _q_ is below a predefined threshold (_epsilon_)
        * Reaching _q_ indirectly means that there is a chain of directly reachable instance between _p_ and _q_
    * Non-core instances can be part of a cluster at its edge
* Pros
    * Can detect clusters of arbitrary shape
    * Robust to outliers
* Cons
    * Computationally more complex than K-means
    * Difficulty to define hyper-parameter values (_delta_ and _epsilon_)
* _DBSCAN_ example:
![DBSCAN example](https://i.imgur.com/MWapikL.png)
* _K-means_ VS _DBSCAN_ output example:
![Comparison](https://i.imgur.com/jqzzyJI.png)

#### __Agglomerative hierarchical clustering__
* Easy-to-understand and interpret
* Creates graph-based clusters, namely dendograms
* More information, see [here](https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/)
* Linkage is done by the smallest value in the distances matrix -> smaller distance value means closer instances
* A new matrix is computed after each aggregation, featuring the newly computed pair
* Created dendrogram depends on the linkage criteria:
    * Single linkage
    * Complete linkage
    * Average linkage
* Pros
    * Easily interpretable, but more confusing for large datasets
    * Setting of hyper-parameter values is easy
* Cons
    * Computationally more complex than K-means
    * Interpretation of dendrograms can be, in several domains, quite subjective
    * Gets often stuck into local optima

####  Evaluation of clusters
* __Silhouette__
    * Evaluates compactness inside clusters
    * Computed for each object x, can return a value:
        * _~1_, meaning x is close to the cluster center
        * _~0_, meaning x is on the boundary of two clusters
        * _~-1_, meaning x should be in other cluster
    * The average over all objects is returned as the silhouette value for the whole clustering 
* __Within-groups sum of squares__
    * Previously presented in _K-means_ fo finding the ideal number of clusters
* __Jaccard index__
    * Also know as Intersection over Union
    * Can be used if objects have labels
    * Measures how well the clusters separate objects regarding their labels

#### Important Notes
* Various methods and even various runs of the same method might return different results
    * _e.g_ initial centroid positions on the _K-means_ algorithm
* Noisy data is harder to process and can reduce the quality of the found solutions