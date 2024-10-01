import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    dv, model, quiz_answers = data

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("homework")

    with mlflow.start_run() as run:
        with open('dict_vectorizer.bin', 'wb') as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact('dict_vectorizer.bin')
        mlflow.sklearn.log_model(model, 'model')

        logged_model = f'runs:/{run.info.run_id}/model'
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        print("Model model_size_bytes:", loaded_model.metadata.model_size_bytes)
        quiz_answers["Q6"] = loaded_model.metadata.model_size_bytes
        mlflow.log_params(quiz_answers)

        return run.info.run_id
