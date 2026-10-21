# Project Status (mid-semester)

## Completed and verified on Tesla T4
- [x] Environment Manager with GPU guard (environment.py)
- [x] Triton kernels: vector add, elementwise multiply, block-sum (kernels.py)
- [x] Config generator with validation and recorded rejections (config_generator.py)
- [x] Correctness verifier with NaN-poisoned outputs and a negative test (verify.py)
- [x] Benchmarking with do_bench, repeats, median/spread, head-to-head baseline comparison (benchmark.py)
- [x] Exhaustive search tuner with attempted/successful/failed counts (tuner.py)
- [x] JSON tuning cache: stable key, hit/miss, versioning, invalidation, corruption recovery (cache.py, cached_tuner.py)

## Written, pending final verification
- [ ] Plots, report and exports (visualization.py, report.py)

## Yet to be done
- [ ] Random search and exhaustive-vs-random comparison
- [ ] Successive halving
- [ ] Bayesian optimization (optional)
- [ ] Command-line interface
- [ ] Full documentation and viva preparation

## Notes
- The best config is the fastest among those tested, not a proven global optimum.
- Several configs are near-tied, so the winner can change between runs.
- Small inputs (~10K elements) are dominated by launch overhead.
