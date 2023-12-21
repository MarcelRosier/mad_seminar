from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import numpy as np
from sklearn.metrics import roc_auc_score


BINS = 200


class EvalType(Enum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    ALL = "all"
    MERGED = "merged"


class GanomalyEvaluator:
    def __init__(self, model, dataloaders) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.merge_abnormal = False

    def evaluate_model(self):
        label_score_dict = {}
        for label, dataloader in self.dataloaders.items():
            anomaly_scores = []

            for batch in dataloader:
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images = batch
                # Assuming your model has a detect_anomaly method
                result = self.model.detect_anomaly(images)
                anomaly_scores.extend(result["anomaly_score"].cpu().detach().numpy())

            label_score_dict[label] = anomaly_scores

        self.label_score_dict = label_score_dict
        self.print_stats_table(label_score_dict)

    def get_merged_abnormal_scores(self):
        abnormal_scores = []
        for k, v in self.label_score_dict.items():
            if k != "normal":
                abnormal_scores.extend(v)
        return abnormal_scores

    def histplot(self, eval_type=EvalType.MERGED):
        if eval_type == EvalType.MERGED:
            # plot normal
            sns.histplot(
                self.label_score_dict["normal"], bins=BINS, kde=True, label="normal"
            )
            sns.histplot(
                self.get_merged_abnormal_scores(), bins=BINS, kde=True, label="abnormal"
            )
        elif eval_type == EvalType.NORMAL:
            sns.histplot(
                self.label_score_dict["normal"], bins=BINS, kde=True, label="normal"
            )
        elif eval_type == EvalType.ABNORMAL:
            for k, v in self.label_score_dict.items():
                if k != "normal":
                    sns.histplot(v, bins=BINS, kde=True, label=k)
        elif eval_type == EvalType.ALL:
            for k, v in self.label_score_dict.items():
                sns.histplot(v, bins=BINS, kde=True, label=k)

        plt.title("Histogram of Anomaly Scores")
        plt.xlim(left=0)
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    # def calc_auroc():
    #     return roc_auc_score(labels, scores)

    def print_stats_table(self, label_score_dict):
        console = Console()

        table = Table(title="Statistics of Anomaly Scores per categroy")
        table.add_column("Pathology", header_style="bold")
        table.add_column("Minimum", justify="right", header_style="bold")
        table.add_column("Maximum", justify="right", header_style="bold")
        table.add_column("Median", justify="right", header_style="bold")
        table.add_column("Mean", justify="right", header_style="bold")
        table.add_column("Variance", justify="right", header_style="bold")
        table.add_column("#Samples", justify="right", header_style="bold")

        if self.merge_abnormal:
            label_score_dict["abnormal"] = []
            for label, scores in label_score_dict.items():
                if label != "normal":
                    label_score_dict["abnormal"].extend(scores)
            # remove all other labels
            label_score_dict = {
                "abnormal": label_score_dict["abnormal"],
                "normal": label_score_dict["normal"],
            }
        for label, scores in label_score_dict.items():
            min_val, max_val, median_val, mean_val, var_val, n = self.calc_stats(
                scores, label, verbose=False
            )
            table.add_row(
                label,
                f"{min_val:.2f}",
                f"{max_val:.2f}",
                f"{median_val:.2f}",
                f"{mean_val:.2f}",
                f"{var_val:.2f}",
                f"{n}",
            )

        console.print(table)

    def calc_stats(self, scores, label, verbose=True):
        # Calculate statistics
        min_val = np.min(scores)
        max_val = np.max(scores)
        median_val = np.median(scores)
        mean_val = np.mean(scores)
        var_val = np.var(scores)
        numberSamples = len(scores)

        if verbose:
            # Print statistics
            print(f"Dataset: {label}")
            print(f"  Minimum: {min_val}")
            print(f"  Maximum: {max_val}")
            print(f"  Median: {median_val}")
            print(f"  Mean: {mean_val}")
            print(f"  Variance: {var_val}")
            print(f"  Number of samples: {numberSamples}")

        return min_val, max_val, median_val, mean_val, var_val, numberSamples
