from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


def get_workspace(
    workspace_name="lab-azure-ml",
    subscription_id="583f14e6-f8a6-4f80-835e-77e5f8d9410a",
    resource_group="lab-azure-ml",
    auth=AzureCliAuthentication()
) -> Workspace:
    """Convenience function for initing the lab workspace. Pass args if you want a different one."""
    return Workspace(
        workspace_name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=auth
    )
