# Data Folder

This project runs out of the box by generating a synthetic SDSS-like astronomy dataset.

If you later download a real SDSS-style dataset, place it here as:

```text
data/sdss_galaxy_data.csv
```

The training script expects these columns:

- `u`, `g`, `r`, `i`, `z`: photometric magnitudes
- `redshift`: redshift value
- `class`: target label, such as `GALAXY`, `STAR`, or `QSO`

The generated dataset is used only for portfolio demonstration and learning. For a research-grade project, replace it with an official SDSS data export.
