# Data

`CMAPSSData/` holds the FD001 subset of NASA's Turbofan Engine Degradation
Simulation (C-MAPSS), included here so the pipeline runs without a download step.

- `train_FD001.txt` — 100 engines run to failure, 20,631 cycles
- `test_FD001.txt`  — 100 engines truncated before failure, 13,096 cycles
- `RUL_FD001.txt`   — true remaining useful life for each test engine

26 space-separated columns: unit number, cycle, three operational settings, then
21 sensor measurements.

The full archive also contains FD002–FD004, which add operating conditions and a
second fault mode. They are not included here; see the "Possible extensions"
section of the top-level README.

Source: NASA Prognostics Data Repository. US Government work, public domain.
