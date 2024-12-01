import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


class OODDetection:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    # Core Utilities
    def calculate_scores(self, loader, scoring_function):
        self.model.eval()
        scores = []
        embeddings = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(self.device)
                output = self.model(data)
                embeddings.append(output.cpu().numpy())
                batch_scores = scoring_function(output)
                scores.extend(batch_scores.cpu().numpy())
        return np.array(scores), np.vstack(embeddings)

    @staticmethod
    def calculate_detection_rates(scores, threshold):
        scores = np.array(scores)  # Ensure scores are NumPy arrays
        in_detection_rate = (scores >= threshold).mean()
        ood_detection_rate = (scores < threshold).mean()
        return in_detection_rate, ood_detection_rate

    @staticmethod
    def plot_score_distributions(in_scores, ood_scores, threshold=None):
        plt.hist(in_scores, bins=50, alpha=0.6,
                 label='In-Distribution', color='blue')
        plt.hist(ood_scores, bins=50, alpha=0.6,
                 label='Out-of-Distribution', color='red')
        if threshold is not None:
            plt.axvline(threshold, color='green',
                        linestyle='--', label='Threshold')
        plt.legend()
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Score Distributions")
        plt.show()

    # Scoring Functions
    @staticmethod
    def max_confidence(output):
        return torch.max(F.softmax(output, dim=1), dim=1)[0]

    @staticmethod
    def energy_score(output):
        return torch.logsumexp(output, dim=1)

    @staticmethod
    def entropy_score(output):
        probabilities = F.softmax(output, dim=1)
        return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    # Threshold-based OOD Detection
    def threshold_ood_detection(self, test_loader, ood_loader, scoring_function, softmax_threshold=0.5):

        in_scores, _ = self.calculate_scores(test_loader, scoring_function)
        ood_scores, _ = self.calculate_scores(ood_loader, scoring_function)

        # Calculate detection rates
        in_detection_rate, ood_detection_rate = self.calculate_detection_rates(
            in_scores, softmax_threshold)

        # Calculate OOD classification breakdown
        # OOD samples classified as ID
        ood_as_id = (ood_scores > softmax_threshold).sum().item()
        ood_as_od = len(ood_scores) - ood_as_id  # OOD samples classified as OD

        ood_as_id_percent = 100.0 * ood_as_id / len(ood_scores)
        ood_as_od_percent = 100.0 * ood_as_od / len(ood_scores)

        # Plot distributions
        self.plot_score_distributions(
            in_scores, ood_scores, threshold=softmax_threshold)

        print(
            f"Threshold OOD Detection: Softmax Threshold = {softmax_threshold:.2f} with scoring function: {scoring_function}")
        print(f"In-Distribution Detection Rate: {in_detection_rate:.2f}")
        print(f"Out-of-Distribution Detection Rate: {ood_detection_rate:.2f}")
        print(f"OOD samples classified as ID: {ood_as_id_percent:.2f}%")
        print(f"OOD samples classified as OD: {ood_as_od_percent:.2f}%")

        return {
            "in_detection_rate": in_detection_rate,
            "ood_detection_rate": ood_detection_rate,
            "ood_as_id_percent": ood_as_id_percent,
            "ood_as_od_percent": ood_as_od_percent,
        }

    # Mahalanobis Distance OOD Detection

    def mahalanobis_ood_detection(self, test_loader, ood_loader, softmax_threshold=0.5):
        _, in_embeddings = self.calculate_scores(
            test_loader, lambda x: x)
        _, ood_embeddings = self.calculate_scores(
            ood_loader, lambda x: x)

        # Compute mean and covariance
        scaler = StandardScaler().fit(in_embeddings)
        in_embeddings_scaled = scaler.transform(in_embeddings)
        mean = np.mean(in_embeddings_scaled, axis=0)
        covariance = np.cov(in_embeddings_scaled, rowvar=False)
        covariance_inv = np.linalg.inv(covariance)

        # Calculate Mahalanobis distances
        in_distances = np.array([mahalanobis(x, mean, covariance_inv)
                                for x in in_embeddings_scaled])
        ood_distances = np.array([mahalanobis(x, mean, covariance_inv)
                                  for x in scaler.transform(ood_embeddings)])

        # Calculate detection rates
        in_detection_rate, ood_detection_rate = self.calculate_detection_rates(
            -in_distances, -softmax_threshold)  # Negate distances for comparison

        # OOD classification breakdown
        ood_as_id = (ood_distances < softmax_threshold).sum()
        ood_as_od = len(ood_distances) - ood_as_id

        ood_as_id_percent = 100.0 * ood_as_id / len(ood_distances)
        ood_as_od_percent = 100.0 * ood_as_od / len(ood_distances)

        # Plot distributions
        self.plot_score_distributions(-in_distances, -
                                      ood_distances, threshold=-softmax_threshold)

        print(
            f"Mahalanobis OOD Detection: Softmax Threshold = {softmax_threshold:.2f}")
        print(f"In-Distribution Detection Rate: {in_detection_rate:.2f}")
        print(f"Out-of-Distribution Detection Rate: {ood_detection_rate:.2f}")
        print(f"OOD samples classified as ID: {ood_as_id_percent:.2f}%")
        print(f"OOD samples classified as OD: {ood_as_od_percent:.2f}%")

        return {
            "in_detection_rate": in_detection_rate,
            "ood_detection_rate": ood_detection_rate,
            "ood_as_id_percent": ood_as_id_percent,
            "ood_as_od_percent": ood_as_od_percent,
        }

    # Logistic Regression OOD Detection
    def logistic_regression_ood_detection(self, test_loader, ood_loader):
        _, in_embeddings = self.calculate_scores(
            test_loader, lambda x: x)
        _, ood_embeddings = self.calculate_scores(
            ood_loader, lambda x: x)

        # Combine data and labels
        X = np.vstack([in_embeddings, ood_embeddings])
        y = np.hstack([np.zeros(len(in_embeddings)),
                       np.ones(len(ood_embeddings))])

        # Train logistic regression
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clf = LogisticRegression(random_state=42).fit(X_scaled, y)

        # Get predictions and probabilities
        predictions = clf.predict(X_scaled)

        # Separate results for in-distribution and out-of-distribution
        in_predictions = predictions[:len(in_embeddings)]
        ood_predictions = predictions[len(in_embeddings):]

        # Calculate metrics
        in_detection_rate = (in_predictions == 0).mean()
        ood_detection_rate = (ood_predictions == 1).mean()
        ood_as_id = (ood_predictions == 0).sum()
        ood_as_od = (ood_predictions == 1).sum()

        ood_as_id_percent = 100.0 * ood_as_id / len(ood_predictions)
        ood_as_od_percent = 100.0 * ood_as_od / len(ood_predictions)

        # Print results
        print(f"Logistic Regression OOD Detection: Decision Threshold = 0.50")
        print(f"In-Distribution Detection Rate: {in_detection_rate:.2f}")
        print(f"Out-of-Distribution Detection Rate: {ood_detection_rate:.2f}")
        print(f"OOD samples classified as ID: {ood_as_id_percent:.2f}%")
        print(f"OOD samples classified as OD: {ood_as_od_percent:.2f}%")

        # Plot distributions
        probabilities = clf.predict_proba(X_scaled)[:, 1]
        in_probs = probabilities[:len(in_embeddings)]
        ood_probs = probabilities[len(in_embeddings):]
        self.plot_score_distributions(
            in_probs,
            ood_probs,
            # title="Logistic Regression Probability Distributions"
        )

        return {
            "in_detection_rate": in_detection_rate,
            "ood_detection_rate": ood_detection_rate,
            "ood_as_id_percent": ood_as_id_percent,
            "ood_as_od_percent": ood_as_od_percent,
            "accuracy": (clf.predict(X_scaled) == y).mean(),
            "logistic_regression_model": clf,
        }

    # Run All Methods
    def run_all_methods(self, test_loader, ood_loader, softmax_threshold=0.5):
        results = {}
        results["threshold"] = self.threshold_ood_detection(
            test_loader, ood_loader, self.max_confidence, softmax_threshold)
        results["threshold"] = self.threshold_ood_detection(
            test_loader, ood_loader, self.energy_score, softmax_threshold)
        results["threshold"] = self.threshold_ood_detection(
            test_loader, ood_loader, self.entropy_score, softmax_threshold)
        # results["mahalanobis"] = self.mahalanobis_ood_detection(
        #     test_loader, ood_loader, softmax_threshold)
        results["logistic_regression"] = self.logistic_regression_ood_detection(
            test_loader, ood_loader)
        return results
