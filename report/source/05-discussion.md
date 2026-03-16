# Discussion

The results in Section IV demonstrate a consistent improvement over existing methods.
We attribute this to two factors.

**Component X** handles edge cases that prior work [@smith2023] left unaddressed.
By explicitly modelling the boundary conditions, the system avoids a class of errors
that accounts for roughly half of the baseline's failures.

**Component Y** acts as a regulariser, preventing over-fitting on the training
distribution. This is particularly important when training data is limited, as is
common in real-world deployments [@brown2021].

## Limitations

The proposed method requires approximately twice the inference time of the baseline.
In latency-sensitive applications this may be a practical constraint. Future work
will explore lightweight approximations that reduce this overhead without sacrificing
accuracy.
