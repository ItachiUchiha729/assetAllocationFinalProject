# Asset Allocation Final Project

Python implementation for the Remote Final Project on factor-return estimation,
costless mean-variance optimization, exact optimization via linear algebra, and
cost-aware optimization.

## Repository Layout

```text
data/       Local data instructions and ignored raw data location
docs/       Assignment prompt and supporting documentation
notebooks/  Analysis notebook and visualizations
src/        Importable project code
```

## Data Setup

The project expects course-provided files named:

```text
pandas-frames.YYYY.pickle.bz2
covariance.YYYY.pickle.bz2
```

Put them in `data/raw/factor_model/`, or set:

```bash
export FACTOR_MODEL_DIR=/path/to/FACTOR_MODEL
```

For the screenshot error, the missing file is `pandas-frames.2003.pickle.bz2`.
Download or copy that file, plus the matching `covariance.2003.pickle.bz2`, into
`data/raw/factor_model/`. For the full project, add every pair for 2003-2010.

```python
from asset_allocation_project import FactorModelData

data = FactorModelData.load(years=range(2003, 2011))
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Open `notebooks/portfolio_visualization.ipynb` and run cells top-to-bottom after
the data files are available. During development, use `years=range(2003, 2004)`
first; for the full assignment use `years=range(2003, 2011)`.
