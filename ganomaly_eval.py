from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)


BINS = 100


class EvalType(Enum):
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    ALL = "all"
    MERGED = "merged"
    TRAIN = "train"


class GanomalyEvaluator:
    def __init__(self, model, dataloaders) -> None:
        self.model = model
        self.dataloaders = dataloaders

    def evaluate_model(self, normalize: bool):
        """
        Evaluate the model on the test set

        Args:
            normalize (bool): normalize the scores
        """
        self.label_score_dict = {}
        self.label_in_rec_dict = {}
        for label, dataloader in self.dataloaders.items():
            anomaly_scores = []
            input_reconstructions_tuples = []
            for batch in dataloader:
                if len(batch) == 3:
                    images, _, _ = batch
                else:
                    images = batch
                # Assuming your model has a detect_anomaly method
                result = self.model.detect_anomaly(images)
                anomaly_scores.extend(result["anomaly_score"].cpu().detach().numpy())
                ## store reconstructions
                input_reconstructions_tuples.extend(
                    zip(
                        images.cpu().detach().numpy(),
                        result["reconstruction"].cpu().detach().numpy(),
                    ),
                )

            self.label_score_dict[label] = anomaly_scores
            self.label_in_rec_dict[label] = input_reconstructions_tuples

        if normalize:
            self.normalize_scores()

    def normalize_scores(self):
        # normalize each array based on the min and max of all arrays
        _, merged_scores = self.get_labeled_scores()
        for k, v in self.label_score_dict.items():
            self.label_score_dict[k] = (v - np.min(merged_scores)) / (
                np.max(merged_scores) - np.min(merged_scores)
            )

    def plot_in_rec(self, label, n=1):
        """
        Plot the input and reconstruction images

        Args:
            label (str): the label to plot the images for
            n (int): the number of images to plot
        """
        input_reconstructions_tuples = self.label_in_rec_dict[label]
        for i in range(n):
            input_img, reconstruction_img = input_reconstructions_tuples[i]
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(input_img.transpose(1, 2, 0), cmap="gray")
            ax[0].set_title("Input")
            ax[1].imshow(reconstruction_img.transpose(1, 2, 0), cmap="gray")
            ax[1].set_title("Reconstruction")
            plt.show()

    def get_merged_abnormal_scores(self):
        abnormal_scores = []
        for k, v in self.label_score_dict.items():
            if k not in ["normal", "train"]:
                abnormal_scores.extend(v)
        return abnormal_scores

    def histplot(self, eval_type=EvalType.MERGED):
        """
        Plot the histogram of anomaly scores

        Args:
            eval_type (EvalType): the type of evaluation to plot the histogram for
        """
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
        elif eval_type == EvalType.TRAIN:
            sns.histplot(
                self.label_score_dict["train"], bins=BINS, kde=True, label="train"
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

    def print_stats_table(self):
        console = Console()

        table = Table(title="Statistics of Anomaly Scores per categroy")
        table.add_column("Pathology", header_style="bold")
        table.add_column("Minimum", justify="right", header_style="bold")
        table.add_column("Maximum", justify="right", header_style="bold")
        table.add_column("Median", justify="right", header_style="bold")
        table.add_column("Mean", justify="right", header_style="bold")
        table.add_column("Variance", justify="right", header_style="bold")
        table.add_column("#Samples", justify="right", header_style="bold")

        for label, scores in self.label_score_dict.items():
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

    # get score labels list
    def get_labeled_scores(self):
        labels = []
        scores = []

        for pathology, anomaly_scores in self.label_score_dict.items():
            if pathology in ["normal"]:
                labels.extend([0] * len(anomaly_scores))
            else:
                labels.extend([1] * len(anomaly_scores))
            scores.extend(anomaly_scores)
        return labels, scores

    def roc_auc_score(self):
        labels, scores = self.get_labeled_scores()
        return roc_auc_score(labels, scores)

    def plot_auroc(self):
        labels, scores = self.get_labeled_scores()

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()

    def plot_auprc(self):
        labels, scores = self.get_labeled_scores()
        precision, recall, _ = precision_recall_curve(labels, scores)
        # print(f"Precision: {precision}, Recall: {recall}")
        auprc = auc(recall, precision)

        plt.figure()
        lw = 2
        plt.plot(
            recall,
            precision,
            color="darkorange",
            lw=lw,
            label=f"PR curve (area = {auprc:.2f})",
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend(loc="lower right")
        plt.show()

    def prc_auc_score(self):
        labels, scores = self.get_labeled_scores()
        precision, recall, _ = precision_recall_curve(labels, scores)
        return auc(recall, precision)

    def plot_confusion_matrix(self, threshold=0.5):
        # Compute confusion matrix
        true_labels, anomaly_scores = self.get_labeled_scores()

        predicted_labels = np.array(anomaly_scores) > threshold
        cm = confusion_matrix(true_labels, predicted_labels)

        # Plot confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal", "Anomalous"],
            yticklabels=["Normal", "Anomalous"],
        )
        plt.title("title")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def find_optimal_threshold_f1(self):
        true_labels, anomaly_scores = self.get_labeled_scores()
        precision, recall, thresholds = precision_recall_curve(
            true_labels, anomaly_scores
        )
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Find the threshold that maximizes the F1-score
        optimal_threshold_index = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_index]

        return optimal_threshold

    def classification_report(self, threshold=0.5):
        # Compute confusion matrix
        true_labels, anomaly_scores = self.get_labeled_scores()

        predicted_labels = np.array(anomaly_scores) > threshold
        print(classification_report(true_labels, predicted_labels))
