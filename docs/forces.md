# Forces

## N-body gravity

The N-body gravity model computes particle accelerations using Newtonian gravity:

```
a_i = sum_{j!=i} G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
```

- `G` is the gravitational constant (settable for normalized units).
- `softening` uses `eps` to avoid singularities at zero separation.
- Self-interaction is excluded (no `i == j` contribution).
- Complexity is O(N^2).

### Softening

Softening is applied consistently in both acceleration and potential energy:

```
PE = -sum_{i<j} G * m_i * m_j / sqrt(|r_ij|^2 + eps^2)
```

### Chunking

For large N, `chunk_size` can be provided to reduce peak memory usage by
processing blocks of particle pairs. This preserves results while trading
some speed for lower memory usage.
