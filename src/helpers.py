# https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/logging/#logging-with-pytorch-lightning
import pytorch_lightning as pl

run = None
try:
    from azureml.core.run import Run, _OfflineRun

    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        run = None
except ImportError:
    print("Couldn't import azureml.core.run.Run")


def get_logger():
    tb_logger = pl.loggers.TensorBoardLogger('logs/')
    logger = [tb_logger]

    if run is not None:
        mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
        mlf_logger = pl.loggers.MLFlowLogger(
            experiment_name=run.experiment.name,
            tracking_uri=mlflow_url,
        )
        mlf_logger._run_id = run.id
        logger.append(mlf_logger)

    return logger
