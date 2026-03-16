# Results

## Quantitative Evaluation

Table I summarises the performance of each method across the three benchmark tasks.
Our approach achieves the highest accuracy on tasks 1 and 3, and matches the
state of the art on task 2.

| Method          | Task 1 (%) | Task 2 (%) | Task 3 (%) |
|:----------------|:----------:|:----------:|:----------:|
| Baseline [@jones2022] | 72.4 | 85.1 | 68.3 |
| Method A [@brown2021] | 78.9 | 87.6 | 74.1 |
| **Ours**        | **83.2**   | **87.5**   | **79.8**   |

The improvement over the baseline is statistically significant ($p < 0.01$,
paired $t$-test). The overall accuracy gain can be expressed as:

$$\Delta = \frac{1}{N} \sum_{i=1}^{N} \left( a_i^{\text{ours}} - a_i^{\text{base}} \right)$$

where $a_i$ denotes the accuracy on task $i$ and $N = 3$.

Fig. 1 shows the per-task accuracy breakdown across all evaluated methods.

![Accuracy comparison across benchmark tasks.](source/figures/placeholder.png)

## Ablation Study

To isolate the contribution of each component, we evaluated three ablated variants.
Removing component X reduces accuracy by 4.1 percentage points, confirming its
importance. Removing component Y has a smaller but consistent effect (−1.8 pp).
