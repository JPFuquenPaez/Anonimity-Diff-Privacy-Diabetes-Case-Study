#  Privacy-Preserving Data Anonymization with k-Anonymity, l-Diversity, and t-Closeness + Case Study

### Privacy-Preserving Data Anonymization with k-Anonymity, l-Diversity, and t-Closeness

This repository explores advanced data anonymization techniques to protect individual privacy in structured datasets. Developed during my Masters in AI and Data Science at Université Paris Dauphine, the project focuses on **k-Anonymity**, **l-Diversity**, and **t-Closeness**, with additional exploration of **Differential Privacy**. The implementation uses Python and the Adult Income Dataset to demonstrate how to balance privacy and utility in real-world datasets.

---

### Core Concepts

#### 1. **k-Anonymity**  
**Objective**: Prevent re-identification by ensuring each group in the dataset contains at least *k* individuals with identical quasi-identifiers.  
**Method**:  
- **Quasi-identifiers** (e.g., age, zip code) are generalized or suppressed.  
- **Mondrian Algorithm**: Greedily partitions data into groups using median splits.  
**Key Result**:  
- Groups of size ≥ *k* reduce re-identification risk but may leak information if sensitive attributes lack diversity.  
**Code Example**:  
```python
def is_k_anonymous(df, partition, k=3):
    return len(partition) >= k
```

#### 2. **l-Diversity**  
**Objective**: Enhance k-Anonymity by ensuring each group has ≥ *l* distinct sensitive attribute values.  
**Method**:  
- Validate partitions using both group size and sensitive attribute diversity.  
**Key Result**:  
- Reduces but does not eliminate probabilistic inference risks (e.g., 4/5 group members share a sensitive value).  
**Code Example**:  
```python
def is_l_diverse(df, partition, sensitive_col, l=2):
    return df.loc[partition, sensitive_col].nunique() >= l
```

#### 3. **t-Closeness**  
**Objective**: Further protect privacy by ensuring sensitive attribute distributions in groups mirror the global dataset.  
**Method**:  
- Measure distribution similarity using **Total Variation Distance** (TVD).  
**Key Result**:  
- Mitigates skewness in sensitive attributes but reduces data utility.  
**Code Example**:  
```python
def is_t_close(df, partition, sensitive_col, global_dist, t=0.2):
    local_dist = df.loc[partition, sensitive_col].value_counts(normalize=True)
    tvd = 0.5 * sum(abs(local_dist.get(c, 0) - global_dist[c]) for c in global_dist)
    return tvd <= t
```

---

### Implementation Highlights

#### Dataset: Adult Income Census Data  
- **Features**: Age, education, occupation, marital status, etc.  
- **Sensitive Attribute**: `income` (binary: ≤50K or >50K).  

#### Key Steps:  
1. **Data Preprocessing**:  
   - Convert categorical features (e.g., `workclass`, `education`) to Pandas categories.  
   - Calculate global spans for numerical features (e.g., `age`, `hours-per-week`).  

2. **Mondrian Partitioning**:  
   - Split data recursively while enforcing k-Anonymity and l-Diversity/t-Closeness.  
   - Visualize partitions as rectangles in 2D feature space:  
   ![Partition Visualization](https://i.imgur.com/partition_rects.png)  

3. **Anonymized Dataset Generation**:  
   - Aggregate quasi-identifiers (e.g., age ranges) and report sensitive attribute distributions.  

---

### Results and Analysis

| Method          | Partitions Created | Privacy Guarantee                          | Utility Trade-off              |
|-----------------|--------------------|--------------------------------------------|---------------------------------|
| **k-Anonymity** | 45                 | Low re-identification risk                 | High utility, low diversity     |
| **l-Diversity** | 68                 | Reduced attribute inference risk           | Moderate utility loss           |
| **t-Closeness** | 92                 | Strong protection against distributional leaks | Significant utility reduction |

**Visual Comparison**:  
![Partition Comparison](https://i.imgur.com/partition_comparison.png)  

- **k-Anonymity**: Large partitions with homogeneous sensitive values.  
- **l-Diversity**: Smaller partitions with forced diversity.  
- **t-Closeness**: Fragmented partitions aligning with global distributions.  

---

### Differential Privacy (Bonus)
Explored a randomization scheme to protect individual entries:  
- **Randomized Response**: With probability *p*, report true income; otherwise, randomize.  
- **Laplace Mechanism**: Add noise to numerical features (e.g., income) for aggregate queries.  

**Key Insight**:  
- Achieves formal privacy guarantees (ε-differential privacy) but introduces noise impacting ML model accuracy.  

---

### Case Study: Healthcare Dataset
Applied techniques to a synthetic healthcare dataset to predict diabetes risk while protecting patient privacy:  
- **Sensitive Columns**: `patient_name`, `ssn` (removed), `has_diabetes` (protected).  
- **Utility**: Logistic regression achieved 72% accuracy on anonymized data (vs 78% on raw data).  

---

### Repository Structure
```
├── Data/                         # Adult and healthcare datasets
├── Notebooks/
│   ├── 1_K_Anonymity.ipynb      # Mondrian algorithm implementation
│   ├── 2_L_Diversity.ipynb      # l-Diversity extension
│   ├── 3_T_Closeness.ipynb      # t-Closeness validation
│   └── 4_Differential_Privacy.ipynb  # Randomized response/Laplace mechanisms
├── References/                  # Key research papers
└── README.md
```

**Tools**: Pandas, NumPy, Matplotlib, Scikit-Learn.  

---

### Challenges & Lessons
1. **Categorical vs. Numerical Handling**: Mondrian’s median splits work poorly for categorical data without encoding.  
2. **Privacy-Utility Trade-off**: t-Closeness often over-fragments data, limiting analytical value.  
3. **Scalability**: NP-hard optimal partitioning limits use on large datasets.  

**Future Work**: Integrate ARX toolkit for optimized anonymization and explore synthetic data generation.  

---

### References
1. [k-Anonymity: A Model for Protecting Privacy](https://epic.org/privacy/reidentification/Sweeney_Article.pdf)  
2. [l-Diversity: Privacy Beyond k-Anonymity](https://personal.utdallas.edu/~muratk/courses/privacy08f_files/ldiversity.pdf)  
3. [t-Closeness: Privacy Beyond k-Anonymity and l-Diversity](https://www.cs.purdue.edu/homes/ninghui/papers/t_closeness_icde07.pdf)  
