import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.cluster import KMeans # Multi-centroid K-means clustering
from sklearn.mixture import GaussianMixture # Multi-centroid GMM clustering
from Model.Centroid import CentroidBasedOneClassClassifier

import logging
import time

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Evaluator(object):
    def __init__(self, model, model_type="autoencoder", metric="AUC", detector_type="cen", n_clusters=3) -> None:
        self.model = model
        self.model_type = model_type
        self.metric = metric
        self.detector_type = detector_type # Detector switch: "cen", "kmeans", or "gmm"
        self.n_clusters = n_clusters # Number of clusters (K) for multi-centroid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_multi_centroid_score(self, train_latent, test_latent, cen_instance): # Compute anomaly score based on detector_type
        actual_k = min(self.n_clusters, len(train_latent)) # Ensure K does not exceed sample count
        if self.detector_type == "kmeans": # Idea 1: K-means multi-centroid mode
            kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10) # Initialize K-means
            kmeans.fit(train_latent) # Fit K centroids on local normal latent data
            dists = kmeans.transform(test_latent) # Compute distances to all K centroids
            return np.min(dists, axis=1) # Return minimum distance as anomaly score
        elif self.detector_type == "gmm": # Idea 1 alternative: GMM mode
            gmm = GaussianMixture(n_components=actual_k, covariance_type="full", random_state=42, reg_covar=1e-5) # Initialize GMM
            gmm.fit(train_latent) # Fit GMM on local normal latent data
            return -gmm.score_samples(test_latent) # Return negative log-likelihood as anomaly score
        else: # Baseline: Original single centroid (CEN) mode
            return cen_instance.get_density(test_latent) # Use original CEN distance calculation

    def calculate_auc(self, y_true, score):
        if not np.all(np.isfinite(score)):
            print("Anomaly score contains infinite or too large values.")
            score = np.nan_to_num(score)  # replace infinities with a large finite number
        
        fpr, tpr, threshold = roc_curve(y_true, score)
        auc_score = auc(fpr, tpr)
        return auc_score
    
    def score_to_label(self, score, threshold):
        return np.where(np.array(score) > threshold, 1, 0)
    
    def calculate_f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred)

    def calculate_precision(self, y_true, y_pred):
        return precision_score(y_true, y_pred)

    def calculate_recall(self, y_true, y_pred):
        return recall_score(y_true, y_pred)

    def calculate_classification_metric(self, y_true, score, threshold=0.5):
        y_pred = self.score_to_label(score, threshold)
        f1 = self.calculate_f1_score(y_true, y_pred)
        precision = self.calculate_precision(y_true, y_pred)
        recall = self.calculate_recall(y_true, y_pred)
        return f1, precision, recall
        
    
    def evaluate(self, test_loader, train_loader=None):
        self.model.eval()
        if self.model_type == "autoencoder":
            anomaly_score = []
            test_label = []
            with torch.no_grad():
                for i, batch_input in zip(tqdm(range(len(test_loader)), desc='Testing batch: ...'), test_loader):
                    _, output, _ = self.model(batch_input[0].to(self.device))
                    recon_loss = torch.nn.MSELoss(reduction="none")(batch_input[0].to(self.device), output)
                    anomaly_score.append(torch.mean(recon_loss, dim=1))
                    test_label.append(batch_input[1])
                anomaly_score = torch.cat(anomaly_score, dim=0).cpu().numpy()
                test_label = torch.cat(test_label, dim=0).cpu().numpy()
                
                if self.metric == "AUC":
                    auc_score = self.calculate_auc(test_label, anomaly_score)
                    logging.info(f"AUC for Autoencoder-based: {auc_score}")
                    return auc_score
                
                if self.metric == "classification":
                    f1, precision, recall = self.calculate_classification_metric(test_label, anomaly_score)
                    logging.info(f"F1 score for Autoencoder-based: {f1}")
                    logging.info(f"Precision for Autoencoder-based: {precision}")
                    logging.info(f"Recall for Autoencoder-based: {recall}")
                    return f1

        if self.model_type == "hybrid":
            anomaly_score = []
            train_latent = []
            with torch.no_grad():
                for i, batch_input in zip(tqdm(range(len(train_loader)), 
                                                desc='Calculate output for training data batch: ...'), train_loader):
                    latent, _, _ = self.model(batch_input[0].to(self.device))
                    train_latent.append(latent)
                train_latent = torch.cat(train_latent, dim=0).cpu().numpy()
                
                test_latent = []
                testing_label = []
                for i, batch_input in zip(tqdm(range(len(test_loader)), desc='Testing batch: ...'), test_loader):
                    latent, _, _ = self.model(batch_input[0].to(self.device))
                    test_latent.append(latent)
                    testing_label.append(batch_input[1])
                    
                test_latent = torch.cat(test_latent, dim=0).cpu().numpy()
                testing_label = torch.cat(testing_label, dim=0).cpu().numpy()
                    
                CEN = CentroidBasedOneClassClassifier()    
                CEN.fit(train_latent)  
                
                if self.metric == "time":
                    start_time = time.time()
                    testing_label = []
                    for i, batch_input in zip(tqdm(range(len(test_loader)), desc='Testing batch: ...'), test_loader):
                        latent, _, _ = self.model(batch_input[0].to(self.device))
                        test_latent.append(latent)
                        testing_label.append(batch_input[1])
                    predictions_cen = self.compute_multi_centroid_score(train_latent, test_latent, CEN) # Calculate score via selected detector
                    end_time = time.time()
                    return end_time - start_time
                    
                
                if self.metric == "AUC":
                    predictions_cen = self.compute_multi_centroid_score(train_latent, test_latent, CEN) # Calculate score via selected detector
                    if not np.all(np.isfinite(predictions_cen)):
                        print("Anomaly score contains infinite or too large values.")
                        predictions_cen = np.nan_to_num(predictions_cen)
                    FPR_cen, TPR_cen, thresholds_cen = roc_curve(testing_label, predictions_cen)
                    cen_auc = auc(FPR_cen, TPR_cen) 
                    logging.info(f"AUC for Hybrid ({self.detector_type}, K={self.n_clusters}) model: {cen_auc}")
                    return cen_auc, test_latent, testing_label

                if self.metric == "classification":
                    predictions_cen = self.compute_multi_centroid_score(train_latent, test_latent, CEN) # Calculate score via selected detector
                    f1, precision, recall = self.calculate_classification_metric(testing_label, predictions_cen)
                    logging.info(f"F1 score for Hybrid model: {f1}")
                    logging.info(f"Precision for Hybrid model: {precision}")
                    logging.info(f"Recall for Hybrid model: {recall}")
                    return f1, test_latent, testing_label
                    
    def visualize(self):
        pass