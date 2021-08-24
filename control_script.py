from azureml.core import Experiment, Environment, ScriptRunConfig, Dataset
from src import helpers

workspace = helpers.get_workspace()

experiment = Experiment(workspace=workspace, name="sabine-test-1")

env = Environment.from_pip_requirements(name="sabine-test-2", file_path="./requirements.txt")
env.docker.enabled = True
env.docker.base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"

dataset = Dataset.get_by_name(workspace, name="convert")
mounted_dataset = dataset.as_mount()

config = ScriptRunConfig(source_directory='./src', compute_target='low-priority-gpu', environment=env,
                         arguments=['--input_data_dir', mounted_dataset],
                         script='test.py')

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)
