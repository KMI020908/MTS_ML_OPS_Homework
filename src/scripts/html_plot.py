import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def update_layout(fig):
     fig.update_layout(
        font=dict(
            size=14,
        )
    )

def prediction_dist(submission):
    fig = px.histogram(
        submission[['preds']].replace({0: 'Stayed', 1: 'Churned'}),
        x='preds',
        title='Prediction distribution',
        width=800,
        height=600,
        histnorm='probability density',
        color_discrete_sequence=['red'],
    )
    update_layout(fig)
    return fig.to_html(full_html=False)

def feature_importance_plot(feature_importance, top_n=5):
    df = pd.DataFrame(
        list(feature_importance.items())[:top_n],
        columns=['feature', 'importance']
    )
    fig = px.bar(
        df,
        x='feature',
        y='importance',
        title=f'Top-{top_n} feature importances',
        width=800,
        height=600,
        color_discrete_sequence=['red']
    )
    update_layout(fig)
    return fig.to_html(full_html=False)

def proba1_dist_plot(submission):
    fig = px.histogram(
        submission,
        x='proba1',
        title='Distribution of the positive class proba',
        width=800,
        height=600,
        histnorm='probability density',
        color_discrete_sequence=['red']
    )
    update_layout(fig)
    return fig.to_html(full_html=False)

def classification_report(report):
    if report is not None:
        classes = list(report.keys())
        for f in ['accuracy', 'macro avg', 'weighted avg']:
            classes.pop(classes.index(f))
        metrics = ['precision', 'recall', 'f1-score']
        data = np.zeros((len(classes), len(metrics)))
        for i, cls in enumerate(classes):
            for j, metric in enumerate(metrics):
                data[i, j] = report[cls][metric]
        heatmap = go.Heatmap(
            z=data,
            x=metrics,
            y=classes,
            colorscale='Inferno'
        )
        layout = go.Layout(
            title='Classification Report Heatmap',
            width=800,
            height=600
        )
        fig = go.Figure(data=[heatmap], layout=layout)
        update_layout(fig)
        fig.update_traces(xgap=2, ygap=2)
        return fig.to_html(full_html=False)
    return None