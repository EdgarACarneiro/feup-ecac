# Summary of ECAC's theoretical material

## Index
* [Clustering](#clustering)
* [Frequent Pattern Mining](#Frequent-Pattern-Mining)
* [Working with Texts](#Working-with-Texts)
* [Recommender systems](#Recommender-systems)
* [Social Network Analysis](#Social-Network-Analysis)
* [Advanced Predictive Methods](#Advanced-Predictive-Methods)
* [Case studies](#Case-studies)

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
Consider the following Transactional data:

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

> Fast Food Transactions were disconsidered since they do not respect the minimum support defined (_0.3_)

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

## Working with Texts

### Text mining tasks
* __Descriptive__
    * Grouping of similar documents
    * Looking for texts about similar issues and words that frequently appear together
* __Predictive__
    * Classification of documents into one or more topics
    * Sentiment analysis and opinion mining
* __Similar to data mining tasks__
    * after transforming the texts into tabular, attribute-value format
* Five phases:
    1. Data acquisition
        * Conversion of text into a sequence of characters
        * Data cleaning of document related info
    2. Feature extraction
    3. Data pre-processing
    4. Model induction
    5. Evaluation and interpretation of results

### Feature extraction
1. __Tokenization__
    * extract, for each text, a sequence of words from the stream of characters
        * each word in the sequence is called a lexical token
    * if a word appears more than once in the text, its token will appear more than once in the sequence of tokens (bag-of-words)
2. __Stemming__
    * Identification of a common base form that represent several variations of a token
    * Avoids having a large number of tokens
    * Transforms tokens into their __stem__
        *_e.g._ for the words “studied”, “studying”, “student”, “studies”, “study” their stem is "studi"
3. __Lemmatization__
    * Sophisticated variation of _stemming_
    * Uses a vocabulary and takes grammatical aspects of language into account, performing a morphological analysis
    * Returns the dictionary form of a word, which is called a __lemma__
![lemma vs stem](https://qph.fs.quoracdn.net/main-qimg-cd7f4bafaa42639deb999b1580bea69f)
4. __Removal of stop words__
    * Reduce the number of stems by removing stop words
        * such as adjectives, adverbs, articles, negations, pronouns, prepositions, conjunctions, qualifiers, etc
    * Stop words that will be removed are application dependent
        *_e.g_ negations are important when mining opinions
5. __Conversion to structured data__
    * Creating a table with binary (presence of a stem in the text) or quantitative (frequency of a stem in the text) values

Text feature extraction example, after applying all steps:
![extraction example](https://www.meaningcloud.com/developer/img/resources/models/models-tokenization-example.png)

### Final remarks
* Most important in text mining is good pre-processing of the text
* A trend is to combine text mining with _NLP_ to get better results

## Recommender systems

### Recommendation task
* Explicit feedback
    * The system asks the user to express a preference on items directly (rating, ranking, etc)
* Implicit feedback
    * Obtained through analysis of user interactions (viewed, saved, copied items, etc)
* Given
    * set of users _{u1, u2, ..., un}_
    * set of items _{i1, i2, ..., i3}_
    * recorded feedbacks _{rui | u in users, i in items}_
* Output
    * model F that predicts for each user-item pair _(u, i)_ a value _rui_ representing the predicted feedback of the user _u_ for the item _i_

* For the examples presented below, consider the following table:

| | Titanic | Pulp Fiction | Iron Man | Forrest Gump | The Mummy |
|:-|:-:|:-:|:-:|:-:|:-:|
| Eve | 1 | 4 | 5 | | 3 |
| Fred | 5 | 1 | | 5 | 2 | 
| Irene | 4 | 1 | 2 | 5 | |
| James | __?__ | 3 | 4 | __?__ | 4 |

* And the respective Item recommendation table

| | Titanic | Pulp Fiction | Iron Man | Forrest Gump | The Mummy |
|:-|:-:|:-:|:-:|:-:|:-:|
| Eve | 1 | 1 | 1 | | 1 |
| Fred | 1 | 1 | | 1 | 1 | 
| Irene | 1 | 1 | 1 | 1 | |
| James | __?__ | 1 | 1 | __?__ | 1 |

### Knowledge based techniques
* Recommendations are based on
    * items' attributes
        * _e.g_ price and type of car, number of seats, etc
    * users' requirements
        * _e.g._ maximum value to pay is _X_, car for family, etc
    * domain knowledge that relates item's attributes with users' requirements
        * _e.g_ family car should have at least 5 seats
    * domain knowledge between users' requirements
        * _e.g._ if family car is required, maximum price must above 2k
* Recommendation process is interactive
* High cost of preparing underlying knowledge, since it is __domain dependent__
    * as a result, __system is inflexible__

### Content based techniques
* Users' interests learned by ML techniques
    * a model of the feedback (Y column) of a given user is learned from the attributes of items rated or ranked by the user in the past
    * Allows prediction of rankings / ratings of users-items pairs

### Model-based collaborative filtering
* Recognizes similarities between users according to their feedback and recommends items preferred by like- minded users
    * ![extras](https://i.imgur.com/1XF3h95.png)
* can provide good results even in cases with low information
![formulas](https://i.imgur.com/X1Rr5WM.png)
* Item recommendation and cosine vector similarity
    * ![example](https://i.imgur.com/QbBdn5j.png)
        * _e.g._ of _sim(x, y)_ calculus (__uses the item recomendation table__)
            * _sim(Eve, Fred) = (1\*1 + 1\*1 + 1\*0 + 0\*1 + 1\*1) / sqrt((1^2 + 1^2 + 1^2 + 0^2 + 1^2) * (1^2 + 1^2 + 0^2 + 1^2 + 1^2) = 3 / sqrt(4*4) = 3 / 4 = 0.75_
            * _sim(James, Irene) = (0\*1 + 1\*1 + 1\*1 + 0\*1 + 1\*0) / sqrt((0^2 + 1^2 + 1^2 + 0^2 + 1^2) * (1^2 + 1^2 + 1^2 + 1^2 + 0^2) = 2 / sqrt(3*4) = 2 / 3.464 = 0.577 =~ 0.58_
        * Notice how Eve, despite having the highest _sim()_ with James, is not considered a nearest neighbour for the movie Forrest Gump -> she did not see the movie, that is why
* Raitng prediction and Pearson correlation similarity
    * ![Imgur](https://i.imgur.com/okKGh1y.png)
> Teacher did not present Pearson formula so expect that not to appear in exam (too complex to)
* Basic idea to map the users and items into a common latent space
* The dimensions of this space, often called factors, represent some implicit properties of items and users’ interests in these implicit properties
* Low-rank factorization
    * user-item matrix ___R___ with _n_ rows and _m_ columns
* Goal
    * user matrix ___W___ of ___n___ rows and _k_ columns
    * item matrix ___H___ of _k_ rows and ___m___ columns
    * such that ___W * H = R___
* Formulas
    * ![formulas](https://i.imgur.com/yXbDyVo.png)
* Matrix factorization example
    * ![Matrix factorization example](https://i.imgur.com/tvZiPWT.png)

### Final remarks
* Measuring the performance of a recommender system is not so easy
    * coverage, scalability, robustness, novelty, serendipity
* Cold start problem arises when a new user or item arrive to the system
    * system cannot draw any inferences for users or items about which it has not yet gathered sufficient information
* Context-based and group recommendations

## Social Network Analysis
* Consider the below figure as an example graph

```
A---B---C
| / | /
D---E
```

* Representation using Graphs
    * undirected
    * weighted
    * directed
    * directed _&_ weighted
* __Adjacency matrix__
    * rows and columns represents nodes
    * if m\[x\]\[y\] > 0, then there X is connected to Y (considering matrix in 1st power)
    * adjacency matrix in 2nd power:
        * shows the number of paths (sequence of edges) of length 2 between pairs of nodes
        * _e.g._ m\[x\]\[y\] = 4, then there are 4 paths of length 2 connecting X to Y
* __Nodes' properties__
    * Degree
        * number of connections of the node
        * in-degree (only makes sense on connected graphs)
            * edges entering in this node
            * in the adjacency matrix: sum of corresponding columns
        * out-degree (only makes sense on connected graphs)
            * edges leaving this node
            * in the adjacency matrix: sum of corresponding rows
    * Distance
        * minimum number of edges that connect two nodes
    * Closeness
        * reflects how accessible a node is in the network
        * decreases with the size of the network
        * _closeness(v) = 1 / sum(distance(u, v), u != v)_
    * Betweenness
        * assesses how important the position of a node in the network is
        * ![Betweenness formula](https://i.imgur.com/8cTwyJZ.png)
        * Betweeness example:
            * ![Betweeness example](https://i.imgur.com/F1T982W.png)
    * Clustering coefficient (of a node)
        * measures the tendency of a node to be included in a triad
        * ![clust_coeff](https://i.imgur.com/wxtXuL1.png)
    * Diameter
        * longest of all distance between the nodes of the network
    * Cliques
        * subset of nodes such that every two nodes in these subsets are connected
        * _e.g_ consider the aboce graph example
            * cliques of size 3 are the following subsets: _{A, B, D}, {B, D, E}, {C, B, E}_
    * Clustering coefficient (of the network)
        * expresses the probability that the triples in the network are connected to form a triangle
        * _e.g._ consider the above graph example
        * triples: _{A, B, C}, {A, B, E}, {A, B, D}, {A, D, E}, {B, C, E}, {B, E, D}, {C, E, D}_
        * triangles: _{A, B, D}, {B, D, E}, {C, B, E}_
        * clustering coefficiente = _3/7_ =~ 0.43
    * Centralization
        * ![centralization](https://i.imgur.com/nb2rTJb.png)
    * Modularity
        * expresses the degree to which a network displays cluster structures (often called communities)

### Final remarks
* Basic node/network properties can be used as features in machine learning applications.
    * _e.g._ link prediction, community detection, etc

---

## Advanced Predictive Methods

### Ensemble learning
* Combine more than one model
    * Class predicted by each base classifier are combined to define the class predicted by the ensemble
* Can be used for predictive and descriptive tasks
    * Classification, regression and clustering
* Main approaches
    * Bagging
    * Boosting
    * Random forest
* Can be:
    * __Homogeneous__
        * All models are induced by the same algorithm
    * __Heterogeneous__
        * Models are induced by different algorithms
* Different combining approaches present in the figure below
    * ![combining approaches](https://i.imgur.com/y4HGS5h.png)
* The combination of the classes predicted by the base classifiers usually occurs by:
    * Voting
    * Weighted voting
    * Stacking

### Bagging
* Parallel approach
* Each base classifier is induced using a sample of the training set
    * The samples, defined using bootstrap, have the same number of objects present in the training set
* Combines the predictions from the base classifiers using voting
* Pros
    * Typically improves the predictive performance of the base learner since the base learner is an unstable predictor
    * Almost hyper-parameter free because the number of trees to be generated does not need to be tuned and decision trees, the most used base learner for bagging, is hyper-parameter free
* Cons
    * Due to the bootstrap sampling, is stochastic (random)
    * Like any ensemble approach, is computationally more expansive than using a single model

### Random Forest
* Combine decision trees, creating a forest-like model
* Like in Bagging, grow each decision tree using a different bootstrap sample
    * Combine individual predictions using voting
    * Different from Bagging, at each node of the tree, only use a predefined number of attributes randomly selected at each node
* Usually has high predictive performance and is interpretable
* Can have a high computational cost
Pros
    * High very good predictive performance in many problems
    * Easy to define/tune its hyper-parameters
* Cons
    * Computationally expensive since the number of recommended trees is large
    * Like bagging, it can be parallelized
    * It is stochastic

### AdaBoost
* Belongs to the Boosting set of methods
    * The idea of boosting is to train weak learners sequentially, each trying to correct its predecessor.
*  At each training iteration, a base classifier is induced using the training set
    *  Each object from the training set is weighted according to how well the base classifier predicted its class
        * The more difficult the correct classification of an object, the larger the weight associated with this object
        * Weight of an object define its probability of being selected for the training of the next base classifier
* Sequential approach
* Pros
    * High predictive performance in several problems
    * Easy to define/tune its hyper-parameter
    * Algorithm with highest performance over kaggle (curiosity)
* Cons
    * Computationally expensive, since the number of generated models depend on the number of iterations
    * Difficult parallelization, since this is a sequential algorithm
    * Difficult interpretation of the induced model

### Algorithm bias
* Definition:
    * Algorithmic bias describes systematic and repeatable errors in a computer system that create unfair outcomes, such as privileging one arbitrary group of users over others
* Each technique is better on a subset of tasks
* Each ML algorithm has an inductive bias
* Preference for selecting one hypothesis over other hypotheses
* Fixed and based on data-independent choices
* Necessary for learning to take place
* Includes:
    * __Declarative__ (representation) __bias__
        * How to represent the hypotheses
        * _e.g._ Prefer conjunctions of predictive attribute values over disjunctions
    * __Procedural__ (search) __Search bias__
        * How to search for the correct hypothesis
        * _e.g._ Start with small hypotheses

### Non-binary classification tasks
* One class classification
* Multiclass classification
* Hierarchical classification
* Multi-label classification
* Multi-target classification
* Ranking classification

### One-class classification
* Training data from just one class (positive) - tries to identify objects of a specific class amongst all objects, by primarily learning from a training set containing only the objects of that class
    * Unary classification
    * Obtain negative class data can be: 
        * Difficult
        * Expensive
        * Time consuming
    * Harm predictive accuracy
    * _e.g._ operation status of a nuclear plant as 'normal'
        * There are few, if any, examples of the negative class
* ![one class classification example](https://i.imgur.com/d1Xm4zW.png)

### Multi-class classification
* Number of possible classes is larger than 2 
    * Many real classification problems are multiclass classification problems
    * Several algorithms deal only with 2 classes
        * To use them, the original multi-class problem is decomposed into several binary classification problems (similar to dummy variables)
* ![multi class classification example](https://i.imgur.com/MQeK45d.png)

### Ranking classification
* Produces output values for N (or less) classes for each input example
    * Ordered by relevance
* Frequently used in information retrieval
    * Search engines
* Error can be measured by ranking comparison
    * True ranking x Predicted ranking
* ![ranking classification example](https://i.imgur.com/PdvuIO0.png)

### Multi-label classification
* Examples can simultaneously belong to more than 1 class
* Transformation into a single-label problem
    * Algorithm independent (Change the data to fit existing algorithms)
        * Combination of conventional single label-classifiers
    * Algorithm dependent (create algorithms to support multi-labels)
        * Modification of single-label classifiers
        * Development of new multi-label algorithms
* ![multi label classification example](https://i.imgur.com/ukpLBv5.png)
* > In multi-class problems the classes are mutually exclusive, whereas for multi-label problems each label represents a different classification task, but the tasks are somehow related (so there is a benefit in tackling them together rather than separately). For example, in the famous leptograspus crabs dataset there are examples of males and females of two colour forms of crab. You could approach this as a multi-class problem with four classes (male-blue, female-blue, male-orange, female-orange) or as a multi-label problem, where one label would be male/female and the other blue/orange. Essentially in multi-label problems a pattern can belong to more than one class.

### Multi-target classification
* More than one target attribute exist
    * Meaning more than one Y variable to predict (_e.g._ Y1 _&_ Y2).
* Of course, it is possible to treat a multi-target classification problem as several independent problems, one per target
* However, the predictions of different target attributes can be somehow correlated
    * Classifier chain is a technique that uses the prediction done for one target attribute as predictive attribute for the remaining target attributes, in a chain-like way
* > They are not same, rather two inter-related concepts but have major difference. Actually multi-label classification is derived from multi-target. In multi-label learning a data instance may be associated with multiple binary class labels. This is as opposed to the traditional task of single-label (i.e., multi-class, or binary) classification where each instance is only associated with a single class label. The multi-label learning problem is in fact a special case of multi-target learning (also known as multi-dimensional or multi-objective), where each label can take multiple values, as opposed to binary labels indicating relevance or not.

###  Hierarchical classification
* Classes can be structured into an hierarchical structure, with subclasses in superclasses
* Each unlabeled object can be classified in:
    * The class associated to any node
        * Prediction to any node
    * One of the leaf nodes of the hierarchical structure
        * Mandatory leaf nodes prediction
* ![hierarchical classification example](https://i.imgur.com/VHM6P5b.png)

### Multiclass problems
* Several ML algorithms induce only binary classifiers
    * _e.g._ SVMs, logistic regression
* What to do?
    * Adapt algorithm to multiclass tasks
    * Use multiclass strategies
        * Decomposition into several binary problems
            * Code matrices
            * Hierarchy of classifiers

### Imbalanced data classification
* Frequent problem in labelled datasets
    * For binary and multiclass tasks
    * Can also occur in regression tasks
* Affect predictive performance of most ML algorithms
* Main approaches to deal with imbalance
    * __Oversampling__
        * Increase number of entries of the minority class
    * __Undersampling__
        * Reduce number of entries of the majority class

###  Incomplete target labelling
* Part of the dataset may have no class label
* But label can be assigned to unlabelled data
* Two main approaches:
    * __Semi-supervised learning__
        * Use labeled objects to assign a class label to the unlabeled objects
    * __Active learning__
        * Select unlabeled objects supposed to be more useful in the learning process
* Humans can also be used for the class labeling - Manual class labeling

### Description _vs_ prediction with supervised interpretable techniques
* Supervised classification techniques can also be used in descriptive tasks
    * They can be trained with all objects in the data set, since the induced model will not be used for prediction
    * Model will describe some patterns present in a data set
        * Allows interpretation of the data available
* The model must be interpretable
    * Excludes black box models

### Final remarks
* Predictive performance can be improved by using ensembles
* Algorithm bias defines how an algorithm fits a dataset
* There are many non-binary classification tasks
* Imbalanced data affects predictive performance
* A dataset can be partially labelled
* Supervised learning techniques can be used for descriptive tasks

## Case studies

### Heterogeneous Ensembles for Long Term Travel Time Prediction
* Results
    * From the four tested predictors, for non-circular lines, the expert-based method and the ensemble approach are the most accurate and robust ones
    * Comparison between these two methods:
        1. The ensemble approach has a small advantage in terms of accuracy;
        2. It is more difficult to include other features for the expert-based method when compared to the ensemble approach;
        3. Much more time and data is necessary to configure the ensemble model when compared to the expert-based method;
        4. Without knowing how to quantify the business value of the increase in accuracy, we can not give a price to pay for a device incorporating these methods on a day-to-day basis in the planning process.

### Bus Bunching Prevention
* Problem Overview
    * In highly frequent bus routes, when one bus falls the schedule, the subsequent bus may pass with a short interval to the first one. This is called an Headway Deviation (HD).
    * Such events often cause Bus Bunching occurrences, the two buses will meet at a bus stop, forming a platoon.
    * The four corrective actions that are used to prevent bus bunching are:
        1. Bus Holding;
        2. Stop-Skipping;
        3. Speed modification;
        4. Short-Turning.
