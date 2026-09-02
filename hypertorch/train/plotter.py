from abc import ABC, abstractmethod
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plotter(ABC):
    """
    Abstract Base Class (ABC) for all experiment plotters in HyperTorch.

    Establishes a common structure, to be inherited by all the future classes
    that specilize in a type of plot (Line, Scatter, etc etc)

    Args:
        experiment_dir: Path to the experiment directory (e.g., 'hypertorch_logs/experiment_0').

    Note: generates the "plot" folder inside the experiment folder, where the plot will be placed.
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.plots_dir = self.experiment_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def plot(self, df: pd.DataFrame, csv_path: Path) -> list[Path]:
        """
        Abstract method that must be implemented by subclasses.

        Args:
            df: Metric DataFrame parsed from the CSV.
            csv_path: Path to the source CSV file.

        Returns:
            A list of Paths pointing to created image files.
        """


class LinePlotter(Plotter):
    """Generates Seaborn line plots for training and evaluation metrics."""

    def __init__(self, experiment_dir: str | Path) -> None:
        super().__init__(experiment_dir)
        match = re.search(r"experiment_(\d+)", self.experiment_dir.name)
        self.num_exp = match.group(1) if match else "0"

    def plot(
        self,
        df: pd.DataFrame,
        csv_path: Path,
        metrics: list[str] | None = None,
    ) -> list[Path]:
        """
        Renders and saves line plots for metrics across epochs/steps.

        Args:
            df: Metric DataFrame returned by LogParser.
            csv_path: Source CSV path returned by LogParser.
            metrics: Optional list of specific base variables to plot (e.g. ['loss']).
                     If None, plots all available variables.

        Returns:
            List of generated plot image file paths.

        Output:
            LinePlot_{variable}_{num_exp}.png
        """
        x_col = (
            "epoch"
            if "epoch" in df.columns
            else ("step" if "step" in df.columns else df.columns[0])
        )

        tracking_cols = {"epoch", "step"}
        metric_cols = [c for c in df.columns if c not in tracking_cols]

        variables = set()
        for col in metric_cols:
            clean_var = col.split("/", 1)[1] if "/" in col else col
            if clean_var not in tracking_cols:
                variables.add(clean_var)

        if metrics:
            variables = {v for v in variables if v in metrics}

        sns.set_theme(style="darkgrid")
        saved_plots: list[Path] = []

        for var_name in sorted(variables):
            matching_cols = [
                c
                for c in metric_cols
                if c == var_name or ("/" in c and c.split("/", 1)[1] == var_name)
            ]

            melted = df.melt(
                id_vars=[x_col],
                value_vars=matching_cols,
                var_name="Split",
                value_name="value",
            ).dropna()

            if melted.empty:
                continue

            melted["Split"] = melted["Split"].apply(
                lambda s: str(s).split("/", 1)[0] if "/" in str(s) else str(s)
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            split_counts = melted["Split"].value_counts()
            single_point_splits = split_counts[split_counts == 1].index.tolist()

            continuous_melted = melted[~melted["Split"].isin(single_point_splits)]
            single_melted = melted[melted["Split"].isin(single_point_splits)]

            for split in single_point_splits:
                val = single_melted[single_melted["Split"] == split]["value"].values[0]
                ax.axhline(
                    y=val,
                    color="#4C72B0" if split == "test" else "gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1,
                    label=f"{split} ({val:.4f})",
                )

            if not continuous_melted.empty:
                sns.lineplot(
                    data=continuous_melted,
                    x=x_col,
                    y="value",
                    hue="Split",
                    marker="o",
                    ax=ax,
                    zorder=3,
                )

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles=handles, labels=labels, title="Split", loc="best")

            formatted_title = var_name.replace("_", " ").capitalize()
            ax.set_title(f"Experiment {self.num_exp} — {formatted_title}")
            ax.set_xlabel(x_col.capitalize())
            ax.set_ylabel(formatted_title)

            clean_filename_var = var_name.replace("/", "_")
            output_filename = f"LinePlot_{clean_filename_var}_{self.num_exp}.png"
            output_path = self.plots_dir / output_filename

            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)

            saved_plots.append(output_path)

        return saved_plots
