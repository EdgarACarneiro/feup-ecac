# Summary of ECAC's theoretical material

---

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


### __K-means__
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

### __DBSCAN__
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

### __Agglomerative hierarchical clustering__
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

###  Evaluation of clusters
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

### Final Remarks
* Various methods and even various runs of the same method might return different results
    * _e.g_ initial centroid positions on the _K-means_ algorithm
* Noisy data is harder to process and can reduce the quality of the found solutions

---

## Frequent Pattern Mining

Nice additional slides on this topic from NANJING University [here](https://cs.nju.edu.cn/zlj/Course/DM_16_Lecture/Lecture_4.pdf).

### Transaction data
Consider the following Transactional data
| TID | Arabic | Indian | Mediterranean | Oriental | Fast Food |
|:-|:-:|:-:|:-:|:-:|:-:|
| Andrew | | X | X | | |
| Bernhard | | X | | X | X |
| Carolina | | X | X | X | |
| Dennis | X | | X | | |
| Eve | | | | X | |
| Fred | | X | X | X | |
| Gwyneth | X | | X | | |
| Hayden | | X | | X | X |
| Irene | | X | X | X | |
| James | X | | X | | |

### Itemsets and their support
* __Itemset__
    * An arbitrary combination of items
    * The number of itemsets grows exponentially with the number of items (naturally)
* __Support of an itemset__
    * The frequency of an itemset in the data
    * Computed as the ratio of transactions containing all items from the itemset to the number of all transactions
        * _e.g_ Support of itemset _{Fast Food}_ is 2/10 = 0.2
        * _e.g_ _Support({Indian, Oriental})_ (intersection) is 5/10 = 0.5

### Frequent itemset mining
* Given
    * A set of all available items
    * Transactional data
    * And a minimum support threshold _min\_sup_
* Goal
    * Find itemsets generated from items that have _Support({I}) > min\_sup_
    * Low _min\_sup_ -> large number of itemsets -> too specific
    * High _min\_sup_ -> small number of itemsets -> too generic

### Itemset lattice
* Represents the search space
![itemset lattice](https://i.imgur.com/Rpc1DRw.png)

### Monotonicity
* If an itemset is frequent then each of its subsets are frequent too
    * _e.g_ _support({Indian, Mediterranean, Oriental})_ = 0.3; _support({Indian, Oriental})_ = 0.4;
    * Make sense since we are "relaxing the restrictions"
* If an itemset is infrequent then none of its supersets will be frequent
    * Opposite of the previous rule
    * Make sense since we are "adding more restrictions"

### Apriori
* Algorithm for frequent item set mining and association rule learning
* Basic Idea: Use the downward closure property to prune the candidate search space
* The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends
* Nice example presented [here](http://www2.cs.uregina.ca/~dbd/cs831/notes/itemsets/itemset_apriori.html)
* Apriori pseudo-code:
![Apriori algorithm](https://i.imgur.com/ItJeplO.png)

### Enumeration tree
* A node exists in the tree corresponding to each frequent itemset
* The root of the tree corresponds to the null itemset
* Let _I = {i1, ... ,1k}_ be a frequent itemset. The parent of the node _I_ is the itemset _{i1, ..., ik-1}_
* Prunes the search space with the frequent itemsets (?)
* Enumeration Tree pseudo-code:
![ET algorithm](https://i.imgur.com/dceZbWj.png)
* Enumeration Tree from the example:
![ET example](https://i.imgur.com/uina5ai.png)

### Eclat
* Vertical data format
    * Intersection of TID-sets for counting the support
    * More efficient than reading each line from the transactional data
![Eclat example](https://i.imgur.com/YrRCmNB.png)

> For unknown reasons, the Fast Food Transactions were disconsidered from now on
> Ask Damas to verify this

### FP-Tree
* Compact representation of transactional data in a tree structure
    * Fast support count and itemset generation
* Only two passes of data required
    1. All frequent items and their support are found
    2. Items in each transaction are processed in a decreasing order according to their support while infrequent items are not considered
* Building an FP-Tree:
![fp-tree build](https://i.imgur.com/bpkCjrG.png)

### FP-Growth
* Division of all frequent itemsets into non-overlapping subsets
> TODO: Ask Damas help

### Maximal and closed frequent itemsets
* Maximal frequent itemset
    * A frequent itemset is maximal if it is frequent and no superset of it is frequent.
* Closed frequent itemset
    * A frequent itemset is closed if it is frequent and no superset of it has the same support.

### __Association rules__
* Rule ___A => C___
    * _A_ and _C_ are non-overlapping itemsets
        * _A_ is the antecedent of the rule
        * _C_ is the consequent of the rule
    * __if-then implication__
        * if the antecedent of the rule is present in some transactions then its consequent should also be present in these transactions
        * in other words, itemsets _A_ and _C_ are associated in the data
    * ___support(A => C) = support(A U C)___
        * _e.g_ _support({Arabic} => {Mediterranean}) = support({Arabic, Mediterranean}) = 0.3 = support({Mediterranean, Arabic})_
    * ___confidence((A => C))__ = support(A => C) / support(A)_
        * Similar to conditional probability

### Association rule mining
* Given
    * A set of all available  items
    * Transactional data
    * Minimum support threshold _min\_sup_
    * Minimum confidence threshold _min\_conf_
* Goal
    * Find association rules generated whose
        * support in the transactional data is at least _min\_sup_
        * confidence in the transactional data is at least _min\_conf_
* Two-phases process:
    1. Mine frequent itemsets with respect to  _min\_sup_ threshold
    2. Generate rules meeting the _min\_conf_ threshold from the found frequent itemsets
* __Monotonicity__
    * _confidence( X => Y - X) >= confidence(X' => Y - X')_
        * With _X'_ being a subset of _X_
        * Moving any item(s) from the antecedent to the consequent does not increase the confidence
            * _e.g_
            * _Y_ = {Indian, Mediterranean, Oriental}
            * _X_ = {Indian, Oriental}
            * _X'_ = {Indian}
            * _Y - X_ = {Indian, Mediterranean, Oriental} - {Indian, Oriental} = {Mediterranean}
            * _Y - X'_ = {Indian, Mediterranean, Oriental} - {Indian} = {Mediterranean, Oriental}
            * _confidence(X => Y - X) = confidence({Indian, Oriental} - {Mediterranean}) = support({Indian, Oriental, Mediterranean}) / support({Indian, Oriental})_ = 0.3 / 0.5 = 0.6
            * _confidence(X' => Y - X') = confidence({Indian} => {Mediterranean, Oriental}) = support({Indian, Mediterranean, Oriental}) / support({Indian})_ = 0.3 / 0.6 = 0.5

### Mining rules from an itemset
Mining rules pseudo-code and example:
![mining rules pseudo-code example](https://i.imgur.com/UqXF9Iy.png)

### Behind support confidence
* Cross-support patterns
    * ___support\_ratio(P)__ = min{s(i1), s(i2), ..., s(ik)} / max{s(i1), s(i2), ..., s(ik)}_
        * _s(i1), s(i2), ..., s(ik)_ are the supports of the tiems _i1, i2, ..., ik_ contained in the pattern _P_
        * _e.g_ _s(i1)_ = 0.9, _s(i2)_ = 0.1, _s(i3)_ = 0.25
        * _support\_ratio({i1 i2})_ = _min{s(i1), s(i2)} / max{s(i1), s(i2)}_ = 0.1 / 0.9 = 0.11
    * __Lift__
        * statistical relationship between the antecedent and the consequent of the rule
        * ___lift(X => Y)__ = confidence(X => Y) / support(Y)_
        * Output
            * _lift(X => Y) > 1_ indicates that the occurrence of _X_ has a positive effect on the occurrence of _Y_
            * _lift(X => Y) < 1_ indicates that the occurrence of _X_ has a negative effect on the occurrence of _Y_
            * _lift(X => Y) ~ 1_ indicates that the occurrence of _X_ has no effect on the occurrence of _Y_

### Simpson's paradox
* Correlations between pairs of itemsets (antecedents and consequents of rules) appearing in different groups of data __may disappear or be reversed when these groups are combined__
    * The relationship observed between the antecedent and the subsequent of the rule can also be influenced by hidden factors that are not captured in the data or not considered in the analysis

Example containing the previous notions exemplified:
![previous slides example](https://i.imgur.com/tggKlGx.png)

### Sequence patterns
* __Event__
    * An itemset pf arbitrary length
* __Sequence__
    * Sequence of events consecutively recorded in time
* __Subsequence__
    * _s1 = <X1, X2, ..., Xn>_ is a subsequence of _s2 = <Y1, Y2, ..., Ym>_ if there exists _1 <= i1 < i2 < ... < in <= m_ such that _X1_ is a subset of _Yi1_, _X2_ a substed of _Yi2_, ..., _Xn_ is a subset of _Yin_

Sequence patterns example:
![Sequence patterns example](https://i.imgur.com/bSfUhRA.png)

### Final Remarks
* Despite the runtime necessities, all patter mining approaches should return the same results
Main obstacle is the large number of resulting patterns
    * Choose _min\_sup_ and _min\_conf_ carefully
    * Other evaluation measures beside support, confidence or lift
    * Measures of “interestingness” of patterns for the users

---

