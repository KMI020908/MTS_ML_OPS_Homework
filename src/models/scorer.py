import json
import xgboost as xgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

class Scorer:

    def get_feature_importance(self):
        self.feature_importance = self.model\
            .get_booster().get_score(importance_type='weight')
        self.feature_importance = {
            k: self.feature_importance[k] for k in sorted(
                self.feature_importance,
                key=self.feature_importance.get,
                reverse=True
            )
        }

    def __init__(self, model_path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.get_feature_importance()
        self.last_submission = None
        self.last_dist_plot = None
        self.last_proba_plot = None
        self.classification_report = None

    def submit(self, client_id, X, y_real=None, threshold=None):
        probas = self.model.predict_proba(X)
        self.get_feature_importance()
        if threshold is None:
            preds = self.model.predict(X)
        else:
            preds = (probas[:, 1] > threshold).astype(int)
        self.last_submission = pd.DataFrame({
            'client_id':  client_id,
            'proba0': probas[:, 0],
            'proba1': probas[:, 1],
            'preds': preds
        })
        if y_real is not None:
            target_names = ['Stayed', 'Churned']
            self.classification_report = classification_report(
                y_real, preds, target_names=target_names, output_dict=True
            )
        return self.last_submission
    
    def save_submission(self, file_path, submission=None):
        if submission is not None:
            submission.to_csv(file_path, index=False)
        elif self.last_submission is not None:
            self.last_submission.to_csv(file_path, index=False)

    def save_feature_importance(self, file_path, top_n=5):
        save_features = self.feature_importance
        if len(self.feature_importance) > top_n:
            save_features = {
                k: self.feature_importance[k]
                for k in list(self.feature_importance.keys())[:top_n]
            }
        with open(file_path, 'w') as json_file:
            json.dump(save_features, json_file)

    def prediction_distribution(self, submission=None, figsize=(10, 6)):
        if submission is None:
            if self.last_submission is not None:
                submission = self.last_submission
            else:
                return
        sns.set_theme(font_scale=1.2, style='darkgrid')
        self.last_dist_plot, ax = plt.subplots(1, figsize=figsize)
        sns.countplot(submission, x='preds', stat='probability', ax=ax)
        ax.set_title('Prediction distribution', fontsize=16)
        plt.close()
        return self.last_dist_plot
    
    def save_prediction_plot(self, file_path, fig=None):
        if fig is not None:
            fig.savefig(file_path, format='png')
        elif self.last_dist_plot is not None:
            self.last_dist_plot.savefig(file_path, format='png')