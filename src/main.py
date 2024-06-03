import os
import json
import pandas as pd
from typing import List
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing_extensions import Annotated
from datetime import datetime
from src.models.preprocessor import Preprocessor
from src.models.scorer import Scorer
from fastapi.templating import Jinja2Templates
from src.scripts import html_plot
from src.models.data_storage_controller import DataStorage


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


with open('config.json') as config_file:
    file_paths = json.load(config_file)['file_paths']

with open(file_paths['preproc_data']) as preproc_data_json:
    preproc_data = json.load(preproc_data_json)

with open('config.json') as config_file:
    storage_params = json.load(config_file)['data_storage']

data_storage_controller = DataStorage(
    storage_params['storage_volume'],
    storage_params['storage_del_frac'],
    storage_params['data_paths']
)
del storage_params

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request}
    )

@app.post("/submission", response_class=HTMLResponse)
async def make_submission(files: Annotated[List[UploadFile], File()], request: Request):
    dtime = str(datetime.now())
    file = files[0]
    df = pd.read_csv(file.file)
    input_path = os.path.join(file_paths['input_folder'], f"input_{dtime}.csv")
    df.to_csv(input_path, index=False)
    train = pd.read_csv(file_paths['train_data'])

    preproc = Preprocessor(**preproc_data).fit(train, train.binary_target)

    scorer = Scorer(file_paths['ML_model'])

    if 'binary_target' in df.columns:
        y_real = df.binary_target
    else:
        y_real = None
    submission = scorer.submit(df.client_id, preproc.transform(df), y_real, threshold=0.5)
    output_path = os.path.join(file_paths['output_folder'], f"output_{dtime}.csv")
    scorer.save_submission(output_path)

    feature_importance_path = os.path\
        .join(file_paths['feature_importance'], f"feature_importance_{dtime}.json")
    scorer.save_feature_importance(feature_importance_path)

    scorer.prediction_distribution()
    plot_path = os.path.join(file_paths['prediction_plot'], f"prediction_plot_{dtime}.png")
    scorer.save_prediction_plot(plot_path)

    pred_plot = html_plot.prediction_dist(submission)
    feature_importance_plot = html_plot.feature_importance_plot(
        scorer.feature_importance
    )
    proba_plot = html_plot.proba1_dist_plot(submission)
    classification_report_plot = html_plot.classification_report(scorer.classification_report)

    data_storage_controller.process_storage()

    context = {
        "request": request,
        "dtime": dtime,
        "pred_plot": pred_plot,
        "proba_plot": proba_plot,
        "feature_importance_plot": feature_importance_plot,
    }
    if classification_report_plot is not None:
        context["classification_report_plot"] = classification_report_plot
    return templates.TemplateResponse(
        "submission.html", context
    )

@app.get("/submission/download")
async def download_submission(dtime: str):
    filepath = os.path.join(file_paths['output_folder'], f"output_{dtime}.csv")
    filename = f'submission_{dtime}.csv'
    if filepath:
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        return FileResponse(filepath, headers=headers, media_type='csv')

@app.get("/input_data/download")
async def download_submission(dtime: str):
    filepath = os.path.join(file_paths['input_folder'], f"input_{dtime}.csv")
    filename = f'input_{dtime}.csv'
    if filepath:
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        return FileResponse(filepath, headers=headers, media_type='csv')

@app.get("/prediction_plot/download")
async def download_submission(dtime: str):
    filepath = os.path.join(file_paths['prediction_plot'], f"prediction_plot_{dtime}.png")
    filename = f'distribution_{dtime}.png'
    if filepath:
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        return FileResponse(filepath, headers=headers, media_type='png')

@app.get("/feature_importance/download")
async def download_submission(dtime: str):
    filepath = os.path.join(file_paths['feature_importance'], f"feature_importance_{dtime}.json")
    filename = f'feature_importance_{dtime}.json'
    if filepath:
        headers = {'Content-Disposition': f'attachment; filename="{filename}"'}
        return FileResponse(filepath, headers=headers, media_type='json')