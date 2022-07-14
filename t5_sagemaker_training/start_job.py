import datetime
from sagemaker.pytorch import PyTorch
import sagemaker
# sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()
# try:
#     role = sagemaker.get_execution_role()
# except ValueError:
#     iam = boto3.client('iam')
#     role = iam.get_role(RoleName='arn:aws:iam::320567679581:role/fsdp-sagemaker-experiments')['Role']['Arn']
# print(role)
smp_config = {
        "ddp": True,
        "tensor_parallel_degree": 8,
        "shard_optimizer_state": True,
        "partitions": 1,
    }
# mpi_options = "-verbose --mca orte_base_help_aggregate 0 "


hyperparameters = {

    "allreduce_post_accumulation": 1,
    "allreduce_post_accumulation_fp16": 1,
}
volume_size = 500
pytorch_estimator = PyTorch(
    role='arn:aws:iam::320567679581:role/fsdp-sagemaker-experiments', # TODO
    #hyperparameters={'model_size': 'large'},
    # entry_point="test_fsdp.py", # the name of the script
    # entry_point="smp_train.py", # the name of the script
    entry_point="main_mp_old.py", # the name of the script
    instance_type="ml.p4d.24xlarge",
    #instance_type="local",
    instance_count=1, # this determines the number of p4d instances
    # image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker-v1.0", 
            # this is a prebuilt PyTorch docker image. Note that you may have to change the path based on the region you are in 
    source_dir="source_dir",
    #output_path='<optional s3 output path>',
    framework_version="1.11.0",
    py_version="py38",
    volume_size=volume_size,
    # dependencies=['source_dir/t5_11/', 'source_dir/t5_11/datasets_grammar/gtrain_1k.csv', '/home/ubuntu/anaconda3/envs/fsdp-sagemaker'],
    region='us-west-2',
    # distribution={
    # "smdistributed": {"modelparallel": {"enabled": True, "parameters": smp_config}},
    # "mpi": {
    #     "enabled": True,
    #     "processes_per_host": 8,
    #     # "custom_mpi_options": mpi_options,
    # },
    # },
    # hyperparameters=hyperparameters,
)
# for i in range(4):
pytorch_estimator.fit(
    job_name='FSDP' + '-' + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
