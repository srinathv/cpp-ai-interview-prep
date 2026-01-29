# Softmax Optimization for AI/ML

## Mathematical Foundation

### Standard Softmax Definition

The softmax function transforms a vector of real numbers into a probability distribution:

    softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j = 1 to N

### The Numerical Stability Problem

Problem: For large values of x, exp(x) overflows (e.g., exp(1000) = infinity)

### Stable Softmax Derivation

Key Insight: Softmax is shift-invariant!

Proof:
    softmax(x - c)_i = exp(x_i - c) / SUM_j exp(x_j - c)
                     = exp(x_i) * exp(-c) / [exp(-c) * SUM_j exp(x_j)]
                     = exp(x_i) / SUM_j exp(x_j)
                     = softmax(x)_i

Solution: Subtract max(x) before computing exp:
    c = max(x)
    softmax(x)_i = exp(x_i - c) / SUM_j exp(x_j - c)

Now all exponents are <= 0, so exp values are in (0, 1].

### Online Softmax (Flash Attention Optimization)

Traditional approach requires 3 passes:
1. Pass 1: Find max(x)
2. Pass 2: Compute sum of exp(x_i - max)
3. Pass 3: Compute softmax values

Online Softmax reduces to 2 passes using incremental updates:
    m_new = max(m_old, x_i)
    d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)

## Optimization Levels

### Level 1: Naive - Simple loop, no numerical stability
### Level 2: Stable - Subtracts max, 3 passes
### Level 3: Online/Fused - Single pass max+sum, shuffle reductions

