"""Factor model project tools.

Utilities for estimating alpha-factor returns and solving the assignment
portfolio-optimization problems.
"""

from __future__ import annotations

import bz2
import os
import pickle
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


INDUSTRY_FACTORS = [
    "AERODEF",
    "AIRLINES",
    "ALUMSTEL",
    "APPAREL",
    "AUTO",
    "BANKS",
    "BEVTOB",
    "BIOLIFE",
    "BLDGPROD",
    "CHEM",
    "CNSTENG",
    "CNSTMACH",
    "CNSTMATL",
    "COMMEQP",
    "COMPELEC",
    "COMSVCS",
    "CONGLOM",
    "CONTAINR",
    "DISTRIB",
    "DIVFIN",
    "DIVYILD",
    "DWNRISK",
    "ELECEQP",
    "ELECUTIL",
    "FOODPROD",
    "FOODRET",
    "GASUTIL",
    "HLTHEQP",
    "HLTHSVCS",
    "HOMEBLDG",
    "HOUSEDUR",
    "INDMACH",
    "INSURNCE",
    "INTERNET",
    "LEISPROD",
    "LEISSVCS",
    "LIFEINS",
    "MEDIA",
    "MGDHLTH",
    "MULTUTIL",
    "OILGSCON",
    "OILGSDRL",
    "OILGSEQP",
    "OILGSEXP",
    "PAPER",
    "PHARMA",
    "PRECMTLS",
    "PSNLPROD",
    "REALEST",
    "RESTAUR",
    "ROADRAIL",
    "SEMICOND",
    "SEMIEQP",
    "SOFTWARE",
    "SPLTYRET",
    "SPTYCHEM",
    "SPTYSTOR",
    "TELECOM",
    "TRADECO",
    "TRANSPRT",
    "WIRELESS",
]

STYLE_FACTORS = ["BETA", "SIZE", "MOMENTUM", "VALUE"]

NON_CANONICAL_RISK_FACTORS = ["LEVERAGE", "LIQUIDTY", "RESVOL", "PROSPECT"]

ALPHA_FACTORS = [
    "1DREVRSL",
    "STREVRSL",
    "LTREVRSL",
    "EARNQLTY",
    "EARNYILD",
    "GROWTH",
    "MGMTQLTY",
    "PROFIT",
    "SEASON",
    "SENTMT",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTOR_MODEL_DIR = PROJECT_ROOT / "data" / "raw" / "factor_model"


def install_pandas_pickle_compatibility() -> None:
    """Add aliases needed to read older pandas pickle files with newer pandas.

    The project data was pickled with an older pandas version that referenced
    classes from pandas.core.indexes.numeric. Newer pandas versions removed that
    private module, but the index data can be safely loaded as regular Index
    objects for this project.
    """
    module_name = "pandas.core.indexes.numeric"
    if module_name in sys.modules:
        return

    numeric_module = types.ModuleType(module_name)
    numeric_module.Int64Index = pd.Index
    numeric_module.UInt64Index = pd.Index
    numeric_module.Float64Index = pd.Index
    numeric_module.NumericIndex = pd.Index
    sys.modules[module_name] = numeric_module


@dataclass
class FactorModelData:
    """Container for factor model frames and covariance matrices."""

    frames: dict[str, pd.DataFrame]
    covariance: dict[str, pd.DataFrame] = field(default_factory=dict)
    model_dir: Path = DEFAULT_FACTOR_MODEL_DIR
    years: range = range(2003, 2004)

    def __post_init__(self) -> None:
        """Validate date ordering and basic data availability."""
        if not self.frames:
            raise ValueError("frames cannot be empty.")
        self.frames = dict(sorted(self.frames.items()))
        missing_required = self.required_columns - set(self.first_frame.columns)
        if missing_required:
            missing = ", ".join(sorted(missing_required))
            raise ValueError(f"the factor data is missing required columns: {missing}")

    @property
    def dates(self) -> list[str]:
        """Return all available model dates."""
        return list(self.frames.keys())

    @property
    def first_frame(self) -> pd.DataFrame:
        """Return the first available cross-sectional frame."""
        return self.frames[self.dates[0]]

    @property
    def canonical_factors(self) -> list[str]:
        """Return the prompt-defined canonical risk factors present in the data."""
        available = set(self.first_frame.columns)
        industry = [factor for factor in INDUSTRY_FACTORS if factor in available]
        style = [factor for factor in STYLE_FACTORS if factor in available]
        return industry + style

    @property
    def available_alpha_factors(self) -> list[str]:
        """Return potential alpha factors present in the data."""
        available = set(self.first_frame.columns)
        return [factor for factor in ALPHA_FACTORS if factor in available]

    @property
    def required_columns(self) -> set[str]:
        """Return columns required for Part 1 estimation."""
        return {"Ret", "IssuerMarketCap"}

    @classmethod
    def load(
        cls,
        model_dir: Optional[Union[str, Path]] = None,
        years: range = range(2003, 2004),
    ) -> "FactorModelData":
        """Load all pandas frame and covariance pickle files from disk."""
        install_pandas_pickle_compatibility()

        model_path = Path(
            model_dir
            or os.environ.get("FACTOR_MODEL_DIR")
            or DEFAULT_FACTOR_MODEL_DIR
        ).expanduser()
        frames: dict[str, pd.DataFrame] = {}
        covariance: dict[str, pd.DataFrame] = {}

        for year in years:
            frame_path = model_path / f"pandas-frames.{year}.pickle.bz2"
            covariance_path = model_path / f"covariance.{year}.pickle.bz2"

            if not frame_path.exists():
                raise FileNotFoundError(
                    f"Missing frame file: {frame_path}\n"
                    "Place the project data under data/raw/factor_model/ or set "
                    "FACTOR_MODEL_DIR to the directory containing "
                    "pandas-frames.YYYY.pickle.bz2 and covariance.YYYY.pickle.bz2."
                )
            if not covariance_path.exists():
                raise FileNotFoundError(
                    f"Missing covariance file: {covariance_path}\n"
                    "Place the project data under data/raw/factor_model/ or set "
                    "FACTOR_MODEL_DIR to the directory containing "
                    "pandas-frames.YYYY.pickle.bz2 and covariance.YYYY.pickle.bz2."
                )

            with bz2.open(frame_path, "rb") as handle:
                frames.update(pickle.load(handle))
            with bz2.open(covariance_path, "rb") as handle:
                covariance.update(pickle.load(handle))

        return cls(frames=frames, covariance=covariance, model_dir=model_path, years=years)


@dataclass
class CostParameters:
    """Transaction-cost inputs for optional cost-aware optimization."""

    spread_bps: float = 0.0
    borrow_bps_annual: float = 0.0
    impact_multiplier: float = 0.0
    trading_days: int = 252

    @property
    def half_spread_rate(self) -> float:
        """Return one-way half-spread cost as a decimal rate."""
        return self.spread_bps / 2.0 / 10_000.0

    @property
    def daily_borrow_rate(self) -> float:
        """Return daily borrow cost as a decimal rate."""
        return self.borrow_bps_annual / 10_000.0 / self.trading_days


@dataclass
class OptimizationResult:
    """Container for out-of-sample optimization results."""

    holdings: dict[pd.Timestamp, pd.Series]
    profit: pd.Series
    transaction_cost: pd.Series
    net_profit: pd.Series
    exposures: pd.DataFrame
    variance_decomposition: pd.DataFrame
    alpha_coefficients: pd.Series


class FactorReturnAnalysis:
    """Part 1 factor return estimation and visualization."""

    def __init__(self, data: FactorModelData, winsor_limit: float = 0.25) -> None:
        self.data = data
        self.winsor_limit = winsor_limit

    @staticmethod
    def winsorize(series: pd.Series, lower: float = -0.25, upper: float = 0.25) -> pd.Series:
        """Clip returns to reduce the effect of extreme outliers."""
        return series.clip(lower=lower, upper=upper)

    def build_estimation_sample(self, frame: pd.DataFrame, alpha_factor: str) -> pd.DataFrame:
        """Build the clean cross-sectional sample for one date and alpha factor."""
        factors = [alpha_factor] + self.data.canonical_factors
        columns = ["Ret", "IssuerMarketCap"] + factors

        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns for {alpha_factor}: {missing}")

        sample = frame.loc[frame["IssuerMarketCap"] > 1e9, columns].copy()
        sample["Ret"] = self.winsorize(
            sample["Ret"],
            lower=-self.winsor_limit,
            upper=self.winsor_limit,
        )
        sample = sample.replace([np.inf, -np.inf], np.nan).dropna()
        return sample

    def estimate_factor_return_for_date(self, frame: pd.DataFrame, alpha_factor: str) -> float:
        """Estimate the alpha factor coefficient for one date with OLS."""
        sample = self.build_estimation_sample(frame, alpha_factor)
        if sample.empty:
            return np.nan

        factor_columns = [alpha_factor] + self.data.canonical_factors
        x = sample[factor_columns].to_numpy(dtype=float)
        y = sample["Ret"].to_numpy(dtype=float)

        # The project statement defines R = X f + error, so no intercept is added.
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        return float(coefficients[0])

    def estimate_alpha_factor_returns(
        self,
        alpha_factors: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Estimate daily return series for all requested potential alpha factors."""
        alpha_factors = alpha_factors or self.data.available_alpha_factors
        factor_returns = pd.DataFrame(index=pd.to_datetime(self.data.dates))

        for alpha_factor in alpha_factors:
            values = []
            for date in self.data.dates:
                values.append(self.estimate_factor_return_for_date(self.data.frames[date], alpha_factor))
            factor_returns[alpha_factor] = values

        return factor_returns.sort_index()

    def cumulative_factor_returns(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative sums of daily factor returns."""
        return factor_returns.cumsum()

    def sharpe_ratios(
        self,
        factor_returns: pd.DataFrame,
        trading_days: int = 252,
    ) -> pd.Series:
        """Compute annualized Sharpe ratios for daily factor return series."""
        return np.sqrt(trading_days) * factor_returns.mean() / factor_returns.std(ddof=1)

    def q1_summary(self, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """Create the Part 1 summary table."""
        summary = pd.DataFrame(
            {
                "Mean Daily Return": factor_returns.mean(),
                "Daily Volatility": factor_returns.std(ddof=1),
                "Annualized Sharpe": self.sharpe_ratios(factor_returns),
                "Cumulative Return": factor_returns.sum(),
                "Observations": factor_returns.count(),
            }
        )
        return summary.sort_values("Annualized Sharpe", ascending=False)

    def plot_cumulative_factor_returns(self, factor_returns: pd.DataFrame, ax: Any = None) -> Any:
        """Plot cumulative sum of estimated factor returns."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(12, 7))[1]
        self.cumulative_factor_returns(factor_returns).plot(ax=ax)
        ax.set_title("Cumulative Estimated Alpha Factor Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative return coefficient")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_sharpe_ratios(self, factor_returns: pd.DataFrame, ax: Any = None) -> Any:
        """Plot annualized Sharpe ratios for estimated factor return series."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(10, 5))[1]
        self.sharpe_ratios(factor_returns).sort_values().plot(kind="barh", ax=ax)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title("Annualized Sharpe Ratios")
        ax.set_xlabel("Sharpe ratio")
        ax.grid(True, axis="x", alpha=0.3)
        return ax


class FactorModelOptimizer(FactorReturnAnalysis):
    """Main project class for factor-return and portfolio optimization."""

    def _require_scipy_optimize(self) -> Any:
        """Import scipy.optimize only when numerical optimization is called."""
        try:
            import scipy.optimize as optimize
        except ImportError as exc:
            raise ImportError(
                "scipy is required for the numerical optimization section. "
                "Install it before running portfolio optimization."
            ) from exc
        return optimize

    def _require_cvxpy(self) -> Any:
        """Import cvxpy only when an optimization method is called."""
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "cvxpy is required for the optimization sections. "
                "Install it before running portfolio optimization."
            ) from exc
        return cp

    def split_dates(self) -> tuple[list[str], list[str]]:
        """Split dates into first 2/3 in-sample and last 1/3 out-of-sample."""
        dates = self.data.dates
        split_index = int(len(dates) * 2 / 3)
        return dates[:split_index], dates[split_index:]

    def estimate_alpha_coefficients(
        self,
        factor_returns: pd.DataFrame,
        in_sample_dates: Optional[list[str]] = None,
    ) -> pd.Series:
        """Estimate the combined-alpha coefficients from the in-sample period."""
        if in_sample_dates is None:
            in_sample_dates, _ = self.split_dates()
        in_sample_index = pd.to_datetime(in_sample_dates)
        return factor_returns.loc[in_sample_index].mean().dropna()

    def clean_optimization_frame(
        self,
        frame: pd.DataFrame,
        alpha_factors: list[str],
        risk_factors: list[str],
    ) -> pd.DataFrame:
        """Keep rows and columns needed for one daily optimization."""
        columns = ["Ret", "IssuerMarketCap", "SpecRisk"] + alpha_factors + risk_factors
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing optimization columns: {missing}")

        clean = frame.loc[frame["IssuerMarketCap"] > 1e9, columns].copy()
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
        clean["Ret"] = self.winsorize(clean["Ret"], -self.winsor_limit, self.winsor_limit)
        return clean

    def combined_alpha_vector(self, frame: pd.DataFrame, alpha_coefficients: pd.Series) -> pd.Series:
        """Compute stock-level combined alpha from alpha loadings and coefficients."""
        aligned = alpha_coefficients.reindex([c for c in alpha_coefficients.index if c in frame.columns])
        alpha_values = frame[aligned.index].to_numpy(dtype=float) @ aligned.to_numpy(dtype=float)
        return pd.Series(alpha_values, index=frame.index, name="CombinedAlpha")

    def square_covariance(self, date: str) -> pd.DataFrame:
        """Return the date covariance as a square DataFrame without renaming labels."""
        if date not in self.data.covariance:
            raise KeyError(f"No covariance matrix found for date {date}.")

        raw = self.data.covariance[date]
        if isinstance(raw, pd.DataFrame) and {"Factor1", "Factor2", "VarCovar"}.issubset(raw.columns):
            matrix = raw.pivot(index="Factor1", columns="Factor2", values="VarCovar")
            matrix = matrix.combine_first(matrix.T)
        elif isinstance(raw, pd.Series):
            if isinstance(raw.index, pd.MultiIndex) and raw.index.nlevels >= 2:
                matrix = raw.unstack()
            else:
                raise ValueError(f"Unsupported covariance Series format for {date}.")
        else:
            matrix = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)

        if isinstance(matrix.index, pd.MultiIndex) and matrix.shape[1] == 1:
            matrix = matrix.iloc[:, 0].unstack()

        if matrix.index.equals(pd.RangeIndex(len(matrix))) and matrix.shape[1] >= 3:
            lower_columns = {str(column).lower(): column for column in matrix.columns}
            factor_1 = next(
                (lower_columns[name] for name in ["factor1", "factor_1", "row", "name1"] if name in lower_columns),
                None,
            )
            factor_2 = next(
                (lower_columns[name] for name in ["factor2", "factor_2", "col", "name2"] if name in lower_columns),
                None,
            )
            value = next(
                (lower_columns[name] for name in ["value", "covariance", "cov", "varcovar"] if name in lower_columns),
                None,
            )
            if factor_1 is not None and factor_2 is not None and value is not None:
                matrix = matrix.pivot(index=factor_1, columns=factor_2, values=value)
                matrix = matrix.combine_first(matrix.T)

        if matrix.index.equals(pd.RangeIndex(len(matrix))) and matrix.shape[0] == matrix.shape[1] + 1:
            labels = matrix.iloc[0].tolist()
            matrix = matrix.iloc[1:].copy()
            matrix.index = labels
            matrix.columns = labels

        if matrix.index.equals(pd.RangeIndex(len(matrix))) and matrix.shape[0] == matrix.shape[1]:
            matrix = matrix.copy()
            matrix.index = matrix.columns

        if matrix.columns.size and matrix.columns[0] not in matrix.index and matrix.shape[0] == matrix.shape[1]:
            first_column = matrix.iloc[:, 0]
            if first_column.dtype == object and first_column.is_unique:
                values = matrix.iloc[:, 1:]
                if values.shape[0] == values.shape[1]:
                    values.index = first_column
                    values.columns = first_column
                    matrix = values

        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        matrix = matrix.combine_first(matrix.T)
        common = [label for label in matrix.index if label in matrix.columns]
        matrix = matrix.loc[common, common].dropna(axis=0, how="all").dropna(axis=1, how="all")
        common = [label for label in matrix.index if label in matrix.columns]
        return matrix.loc[common, common]

    def risk_factors_for_date(self, date: str, frame: Optional[pd.DataFrame] = None) -> list[str]:
        """Return risk factors available in both the date frame and covariance."""
        frame = self.data.frames[date] if frame is None else frame
        covariance = self.square_covariance(date)
        excluded = (
            set(self.data.available_alpha_factors)
            | set(NON_CANONICAL_RISK_FACTORS)
            | {"ID", "Ret", "SpecRisk", "IssuerMarketCap", "DataDate"}
        )
        canonical = set(self.data.canonical_factors)
        return [
            factor
            for factor in covariance.index
            if factor in covariance.columns
            and factor in frame.columns
            and factor in canonical
            and factor not in excluded
            and pd.api.types.is_numeric_dtype(frame[factor])
        ]

    def optimization_diagnostics(self, date: str) -> dict[str, Any]:
        """Return shapes and matched labels used by the optimizer for one date."""
        frame = self.data.frames[date]
        covariance = self.square_covariance(date)
        risk_factors = self.risk_factors_for_date(date, frame)
        return {
            "date": date,
            "frame_shape": frame.shape,
            "covariance_shape": covariance.shape,
            "risk_factor_count": len(risk_factors),
            "first_risk_factors": risk_factors[:20],
            "first_frame_columns": list(frame.columns[:20]),
            "first_covariance_labels": list(covariance.index[:20]),
        }

    def canonical_covariance_matrix(self, date: str, annualized: bool = False) -> pd.DataFrame:
        """Subset the covariance matrix to factors present in the data.

        The raw covariance files are commonly annualized and stored in percent
        units. This method converts them to daily decimal variance by default.
        """
        covariance = self.square_covariance(date)
        matched = self.risk_factors_for_date(date)

        if not matched:
            covariance_labels = [str(label) for label in list(covariance.index[:10])]
            raise ValueError(
                "No risk factors were found in both the date frame and covariance matrix for "
                f"{date}. First covariance labels: {covariance_labels}. "
                f"First frame columns: {[str(label) for label in list(self.data.frames[date].columns[:20])]}"
            )

        matrix = covariance.loc[matched, matched].astype(float).copy()

        if annualized:
            return matrix
        if np.nanmedian(np.abs(matrix.to_numpy())) > 1.0:
            matrix = matrix / 10_000.0
        return matrix / 252.0

    def specific_variance_vector(self, spec_risk: pd.Series) -> np.ndarray:
        """Convert specific risk to daily decimal variance."""
        values = spec_risk.astype(float).to_numpy()
        if np.nanmedian(np.abs(values)) > 1.0:
            values = values / 100.0
        return (values**2) / 252.0

    @staticmethod
    def matrix_square_root(matrix: np.ndarray) -> np.ndarray:
        """Compute a symmetric square root for a positive semidefinite matrix."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    def optimize_one_date(
        self,
        date: str,
        alpha_coefficients: pd.Series,
        kappa: float = 1e-6,
        costs: Optional[CostParameters] = None,
        previous_holdings: Optional[pd.Series] = None,
        initial_holdings: Optional[pd.Series] = None,
        solver: Optional[str] = None,
    ) -> pd.Series:
        """Solve one daily optimization.

        The costless case uses scipy L-BFGS-B with the exact gradient requested
        in the prompt. The cost-aware case remains a convex cvxpy problem
        because the spread, borrow, and impact terms are nonsmooth/nonquadratic.
        """
        costs = costs or CostParameters()

        alpha_factors = list(alpha_coefficients.index)
        risk_factors = self.risk_factors_for_date(date)
        if not risk_factors:
            raise ValueError(f"No usable risk factors found for optimization date {date}.")

        frame = self.clean_optimization_frame(self.data.frames[date], alpha_factors, risk_factors)
        if frame.empty:
            return pd.Series(dtype=float, name=pd.to_datetime(date))

        alpha_vec = self.combined_alpha_vector(frame, alpha_coefficients).to_numpy(dtype=float)
        fvar = self.canonical_covariance_matrix(date).loc[risk_factors, risk_factors]
        x = frame[risk_factors].to_numpy(dtype=float)
        f_sqrt = self.matrix_square_root(fvar.to_numpy(dtype=float))
        q = f_sqrt @ x.T
        spec_var = self.specific_variance_vector(frame["SpecRisk"])
        n_assets = len(frame)

        has_costs = (
            costs.half_spread_rate > 0.0
            or costs.impact_multiplier > 0.0
            or costs.daily_borrow_rate > 0.0
        )
        if not has_costs:
            optimize = self._require_scipy_optimize()
            h0 = np.zeros(n_assets)
            if initial_holdings is not None and not initial_holdings.empty:
                h0 = initial_holdings.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)

            def objective(h: np.ndarray) -> float:
                qh = q @ h
                return float(
                    0.5 * kappa * np.dot(qh, qh)
                    + 0.5 * kappa * np.dot(h * h, spec_var)
                    - np.dot(alpha_vec, h)
                )

            def gradient(h: np.ndarray) -> np.ndarray:
                return kappa * (q.T @ (q @ h) + spec_var * h) - alpha_vec

            result = optimize.fmin_l_bfgs_b(objective, h0, fprime=gradient)
            values = result[0]
            return pd.Series(values, index=frame.index, name=pd.to_datetime(date))

        cp = self._require_cvxpy()
        holdings = cp.Variable(n_assets)
        if initial_holdings is not None and not initial_holdings.empty:
            holdings.value = initial_holdings.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)

        objective = (
            0.5 * kappa * cp.sum_squares(q @ holdings)
            + 0.5 * kappa * cp.sum(cp.multiply(spec_var, cp.square(holdings)))
            - alpha_vec @ holdings
        )

        if costs.half_spread_rate > 0.0 or costs.impact_multiplier > 0.0:
            previous = np.zeros(n_assets)
            if previous_holdings is not None and not previous_holdings.empty:
                previous = previous_holdings.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)
            trade = holdings - previous
            objective += costs.half_spread_rate * cp.norm1(trade)
            if costs.impact_multiplier > 0.0 and "CompositeVolume" in self.data.frames[date].columns:
                volume = self.data.frames[date].reindex(frame.index)["CompositeVolume"].fillna(0.0).to_numpy(dtype=float)
                scale = 1.0 / np.sqrt(np.maximum(volume, 1.0))
                objective += costs.impact_multiplier * cp.sum(cp.multiply(scale, cp.power(cp.abs(trade), 1.5)))

        if costs.daily_borrow_rate > 0.0:
            objective += costs.daily_borrow_rate * cp.sum(cp.pos(-holdings))

        problem = cp.Problem(cp.Minimize(objective))
        solve_kwargs = {"warm_start": True, "verbose": False}
        if solver is not None:
            solve_kwargs["solver"] = solver
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
            problem.solve(**solve_kwargs)
        acceptable_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if problem.status not in acceptable_statuses:
            raise RuntimeError(
                f"cvxpy failed for {date}: status={problem.status}, "
                f"solver={solver or 'default'}"
            )
        if holdings.value is None:
            raise RuntimeError(f"cvxpy did not return a solution for {date}.")

        return pd.Series(holdings.value, index=frame.index, name=pd.to_datetime(date))

    def exact_minimizer_one_date(
        self,
        date: str,
        alpha_coefficients: pd.Series,
        kappa: float = 1e-6,
    ) -> pd.Series:
        """Compute the exact costless minimizer using Woodbury factorization."""
        alpha_factors = list(alpha_coefficients.index)
        risk_factors = self.risk_factors_for_date(date)
        if not risk_factors:
            raise ValueError(f"No usable risk factors found for optimization date {date}.")

        frame = self.clean_optimization_frame(self.data.frames[date], alpha_factors, risk_factors)
        if frame.empty:
            return pd.Series(dtype=float, name=pd.to_datetime(date))

        alpha_vec = self.combined_alpha_vector(frame, alpha_coefficients).to_numpy(dtype=float)
        x = frame[risk_factors].to_numpy(dtype=float)
        fvar = self.canonical_covariance_matrix(date).loc[risk_factors, risk_factors].to_numpy(dtype=float)
        spec_var = self.specific_variance_vector(frame["SpecRisk"])

        spec_var = np.maximum(spec_var, 1e-12)
        d_inv = 1.0 / spec_var
        d_inv_alpha = d_inv * alpha_vec
        d_inv_x = d_inv[:, None] * x

        try:
            f_inv = np.linalg.inv(fvar)
        except np.linalg.LinAlgError:
            f_inv = np.linalg.pinv(fvar)

        middle = f_inv + x.T @ d_inv_x
        rhs = x.T @ d_inv_alpha
        try:
            correction = np.linalg.solve(middle, rhs)
        except np.linalg.LinAlgError:
            correction = np.linalg.pinv(middle) @ rhs

        holdings = (d_inv_alpha - d_inv_x @ correction) / kappa
        return pd.Series(holdings, index=frame.index, name=pd.to_datetime(date))

    def transaction_cost_for_date(
        self,
        date: str,
        holdings: pd.Series,
        previous_holdings: Optional[pd.Series],
        costs: CostParameters,
    ) -> float:
        """Estimate spread, impact, and borrow costs for one date."""
        previous = pd.Series(0.0, index=holdings.index)
        if previous_holdings is not None and not previous_holdings.empty:
            previous = previous_holdings.reindex(holdings.index).fillna(0.0)

        trade = holdings - previous
        total_cost = costs.half_spread_rate * float(np.abs(trade).sum())

        if costs.impact_multiplier > 0.0 and "CompositeVolume" in self.data.frames[date].columns:
            volume = self.data.frames[date].reindex(holdings.index)["CompositeVolume"].fillna(0.0).to_numpy(dtype=float)
            scale = 1.0 / np.sqrt(np.maximum(volume, 1.0))
            total_cost += costs.impact_multiplier * float(np.sum(scale * np.abs(trade.to_numpy(dtype=float)) ** 1.5))

        if costs.daily_borrow_rate > 0.0:
            total_cost += costs.daily_borrow_rate * float(np.clip(-holdings.to_numpy(dtype=float), 0.0, None).sum())

        return total_cost

    def portfolio_profit(self, date: str, holdings: pd.Series) -> float:
        """Compute daily pre-transaction-cost profit h dot R."""
        returns = self.data.frames[date].reindex(holdings.index)["Ret"].astype(float)
        returns = self.winsorize(returns, -self.winsor_limit, self.winsor_limit)
        return float(np.dot(holdings.to_numpy(dtype=float), returns.to_numpy(dtype=float)))

    def exposure_summary(self, holdings: pd.Series) -> pd.Series:
        """Compute long, short, net, and gross market values."""
        values = holdings.to_numpy(dtype=float)
        return pd.Series(
            {
                "long": values[values > 0.0].sum(),
                "short": values[values < 0.0].sum(),
                "net": values.sum(),
                "gross": np.abs(values).sum(),
            },
            name=holdings.name,
        )

    def variance_decomposition(self, date: str, holdings: pd.Series) -> pd.Series:
        """Decompose total variance into idiosyncratic, industry, and style shares.

        The covariance matrix can contain industry-style cross-covariances. To
        make the three reported fractions add to one, the cross term is split
        equally between the industry and style buckets.
        """
        frame = self.data.frames[date].reindex(holdings.index)
        h = holdings.to_numpy(dtype=float)

        fvar = self.canonical_covariance_matrix(date)
        canonical_factors = [factor for factor in fvar.index if factor in frame.columns]
        fvar = fvar.loc[canonical_factors, canonical_factors]
        x_all = frame[canonical_factors].to_numpy(dtype=float)
        exposure_all = x_all.T @ h
        fvar_all = fvar.to_numpy(dtype=float)
        factor_variance = float(exposure_all @ fvar_all @ exposure_all)

        industry_factors = [
            f
            for f in fvar.index
            if f in frame.columns and f not in STYLE_FACTORS
        ]
        style_factors = [f for f in STYLE_FACTORS if f in fvar.index and f in frame.columns]

        industry_variance_standalone = 0.0
        if industry_factors:
            x_industry = frame[industry_factors].to_numpy(dtype=float)
            f_industry = fvar.loc[industry_factors, industry_factors].to_numpy(dtype=float)
            exp_industry = x_industry.T @ h
            industry_variance_standalone = float(exp_industry @ f_industry @ exp_industry)

        style_variance_standalone = 0.0
        if style_factors:
            x_style = frame[style_factors].to_numpy(dtype=float)
            f_style = fvar.loc[style_factors, style_factors].to_numpy(dtype=float)
            exp_style = x_style.T @ h
            style_variance_standalone = float(exp_style @ f_style @ exp_style)

        cross_variance = factor_variance - industry_variance_standalone - style_variance_standalone
        industry_variance = industry_variance_standalone + 0.5 * cross_variance
        style_variance = style_variance_standalone + 0.5 * cross_variance

        spec_var = self.specific_variance_vector(frame["SpecRisk"])
        idiosyncratic_variance = float(np.dot(h**2, spec_var))
        total_variance = factor_variance + idiosyncratic_variance
        total_variance = max(total_variance, 0.0)

        if total_variance == 0.0:
            idio_fraction = industry_fraction = style_fraction = np.nan
        else:
            idio_fraction = idiosyncratic_variance / total_variance
            industry_fraction = industry_variance / total_variance
            style_fraction = style_variance / total_variance

        return pd.Series(
            {
                "total_vol": np.sqrt(total_variance),
                "idiosyncratic_fraction": idio_fraction,
                "industry_fraction": industry_fraction,
                "style_fraction": style_fraction,
                "fraction_sum": idio_fraction + industry_fraction + style_fraction,
                "factor_variance": factor_variance,
                "industry_variance": industry_variance,
                "style_variance": style_variance,
                "industry_variance_standalone": industry_variance_standalone,
                "style_variance_standalone": style_variance_standalone,
                "industry_style_cross_variance": cross_variance,
                "idiosyncratic_variance": idiosyncratic_variance,
                "total_variance": total_variance,
            },
            name=pd.to_datetime(date),
        )

    def run_out_of_sample_optimization(
        self,
        factor_returns: pd.DataFrame,
        kappa: float = 1e-6,
        costs: Optional[CostParameters] = None,
        solver: Optional[str] = None,
        max_dates: Optional[int] = None,
        use_costless_initial_holdings: bool = False,
    ) -> OptimizationResult:
        """Run Problem 3.2 optimization over the out-of-sample period."""
        in_sample_dates, out_of_sample_dates = self.split_dates()
        if max_dates is not None:
            out_of_sample_dates = out_of_sample_dates[:max_dates]
        alpha_coefficients = self.estimate_alpha_coefficients(factor_returns, in_sample_dates)

        holdings_by_date: dict[pd.Timestamp, pd.Series] = {}
        profits: dict[pd.Timestamp, float] = {}
        transaction_costs: dict[pd.Timestamp, float] = {}
        net_profits: dict[pd.Timestamp, float] = {}
        exposures = []
        variance_rows = []
        previous_holdings: Optional[pd.Series] = None
        costs = costs or CostParameters()

        for date in out_of_sample_dates:
            holdings = self.optimize_one_date(
                date=date,
                alpha_coefficients=alpha_coefficients,
                kappa=kappa,
                costs=costs,
                previous_holdings=previous_holdings,
                initial_holdings=(
                    self.exact_minimizer_one_date(date, alpha_coefficients, kappa)
                    if use_costless_initial_holdings
                    else None
                ),
                solver=solver,
            )
            timestamp = pd.to_datetime(date)
            holdings_by_date[timestamp] = holdings
            profits[timestamp] = self.portfolio_profit(date, holdings)
            transaction_costs[timestamp] = self.transaction_cost_for_date(date, holdings, previous_holdings, costs)
            net_profits[timestamp] = profits[timestamp] - transaction_costs[timestamp]
            exposures.append(self.exposure_summary(holdings))
            variance_rows.append(self.variance_decomposition(date, holdings))
            previous_holdings = holdings

        return OptimizationResult(
            holdings=holdings_by_date,
            profit=pd.Series(profits, name="profit").sort_index(),
            transaction_cost=pd.Series(transaction_costs, name="transaction_cost").sort_index(),
            net_profit=pd.Series(net_profits, name="net_profit").sort_index(),
            exposures=pd.DataFrame(exposures).sort_index(),
            variance_decomposition=pd.DataFrame(variance_rows).sort_index(),
            alpha_coefficients=alpha_coefficients,
        )

    def run_exact_out_of_sample_optimization(
        self,
        factor_returns: pd.DataFrame,
        kappa: float = 1e-6,
        max_dates: Optional[int] = None,
    ) -> OptimizationResult:
        """Run the exact costless minimizer from Problem 3.3 over out-of-sample dates."""
        in_sample_dates, out_of_sample_dates = self.split_dates()
        if max_dates is not None:
            out_of_sample_dates = out_of_sample_dates[:max_dates]
        alpha_coefficients = self.estimate_alpha_coefficients(factor_returns, in_sample_dates)

        holdings_by_date: dict[pd.Timestamp, pd.Series] = {}
        profits: dict[pd.Timestamp, float] = {}
        exposures = []
        variance_rows = []

        for date in out_of_sample_dates:
            holdings = self.exact_minimizer_one_date(
                date=date,
                alpha_coefficients=alpha_coefficients,
                kappa=kappa,
            )
            timestamp = pd.to_datetime(date)
            holdings_by_date[timestamp] = holdings
            profits[timestamp] = self.portfolio_profit(date, holdings)
            exposures.append(self.exposure_summary(holdings))
            variance_rows.append(self.variance_decomposition(date, holdings))

        profit = pd.Series(profits, name="profit").sort_index()
        zero_cost = pd.Series(0.0, index=profit.index, name="transaction_cost")
        return OptimizationResult(
            holdings=holdings_by_date,
            profit=profit,
            transaction_cost=zero_cost,
            net_profit=profit.rename("net_profit"),
            exposures=pd.DataFrame(exposures).sort_index(),
            variance_decomposition=pd.DataFrame(variance_rows).sort_index(),
            alpha_coefficients=alpha_coefficients,
        )

    def plot_cumulative_profit(self, profit: pd.Series, ax: Any = None) -> Any:
        """Plot cumulative out-of-sample profit."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(12, 6))[1]
        profit.cumsum().plot(ax=ax)
        ax.set_title("Out-of-Sample Cumulative Profit")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative dollars")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_variance_decomposition(self, decomposition: pd.DataFrame, ax: Any = None) -> Any:
        """Plot idiosyncratic, industry, and style variance fractions."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(12, 6))[1]
        columns = ["idiosyncratic_fraction", "industry_fraction", "style_fraction"]
        decomposition[columns].plot(ax=ax)
        ax.set_title("Variance Decomposition")
        ax.set_xlabel("Date")
        ax.set_ylabel("Fraction of total variance")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_exposures(self, exposures: pd.DataFrame, ax: Any = None) -> Any:
        """Plot long, short, net, and gross market values."""
        import matplotlib.pyplot as plt

        ax = ax or plt.subplots(figsize=(12, 6))[1]
        exposures[["long", "short", "net", "gross"]].plot(ax=ax)
        ax.set_title("Optimal Portfolio Market Values")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dollars")
        ax.grid(True, alpha=0.3)
        return ax
