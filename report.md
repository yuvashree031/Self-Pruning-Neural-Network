# Self-Pruning Neural Network Report

## L1 Penalty and Sparsity

An L1 penalty on the sigmoid gates encourages sparsity because the L1 norm provides a constant gradient magnitude towards zero, regardless of how small the value is. In contrast to L2 regularization, which applies a progressively smaller penalty as values approach zero (causing weights to become small but rarely exactly zero), L1 regularization drives parameters exactly to zero. When applied to the positive values produced by the sigmoid activation function, the L1 penalty effectively pushes the gate values towards the lower limit, thereby pruning the associated connections.

## Sparsity vs. Accuracy Trade-off

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|--------------|
| 0.001  | 54.12        | 4.31         |
| 0.01   | 54.33        | 23.53        |
| 0.1    | 54.17        | 37.88        |

The results demonstrate the trade-off between model sparsity and classification accuracy. As the regularization parameter (Lambda) increases, the sparsity level increases significantly, meaning more weights are pruned. Despite the substantial increase in sparsity, the test accuracy remains relatively stable, suggesting that the self-pruning mechanism successfully identifies and removes less important connections without severely degrading performance.

## Final Gate Values Distribution

The distribution of the final gate values typically exhibits a bimodal pattern. A large spike at zero indicates the successfully pruned weights, while another cluster of values near one represents the connections that the network has determined are crucial for the classification task.

![Gate Value Distribution](gate_distribution.png)
