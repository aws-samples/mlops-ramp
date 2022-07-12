
import json
import boto3
import os
from time import strftime
import subprocess
import sagemaker

import datetime
import glob
import os
import time
import warnings

from sagemaker.pytorch import PyTorch

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

import base64
from botocore.exceptions import ClientError


instance_type = os.environ["INSTANCE_TYPE"]


def lambda_handler(event, context):
    role = 'arn:aws:iam::806174985048:role/service-role/AmazonSageMaker-ExecutionRole-20201218T151409'
    # role = 'arn:aws:iam::${account_id}:role/service-role/AmazonSageMaker-ExecutionRole-{}'          ### <== 1. Role 추가
    
    sagemaker_session = sagemaker.Session()
    
    experiment_name = 'yolov5-poc-exp1'                                                             ### <== 2. Experiment 명
    
    instance_count = 1
    do_spot_training = False
    max_wait = None
    max_run = 1*60*60
    
    ## SageMaker Experiments Setting
    try:
        sm_experiment = Experiment.load(experiment_name)
    except:
        sm_experiment = Experiment.create(
            experiment_name=experiment_name,
            tags=[{'Key': 'model-name', 'Value': 'yolov5'}]
        )    
    
    ## Trials Setting
    create_date = strftime("%m%d-%H%M%s")    
    spot = 's' if do_spot_training else 'd'
    i_tag = instance_type.replace(".","-")
    trial = "-".join([i_tag,str(instance_count),spot])
       
    sm_trial = Trial.create(trial_name=f'{experiment_name}-{trial}-{create_date}',
                            experiment_name=experiment_name)

    job_name = f'{sm_trial.trial_name}'
    
    
    bucket = 'yudong-data'                                                             ### <== 3. 사용할 Bucket 명
    code_location = f's3://{bucket}/yolov5/sm_codes'
    output_path = f's3://{bucket}/yolov5/output' 
    s3_log_path = f's3://{bucket}/yolov5/tf_logs'
    
    hyperparameters = {
        'data': 'data.yaml',
        'cfg': 'yolov5s.yaml',
        'weights': 'weights/yolov5s.pt',
        'batch-size': 128,
        'epochs': 3,
        'project': '/opt/ml/model',
        'workers': 8,
        'freeze': 10
    }
    
    
    s3_data_path = f's3://{bucket}/yolov5/BCCD'
    checkpoint_s3_uri = f's3://{bucket}/poc_yolov5/checkpoints'
    
    distribution = {}

    if hyperparameters.get('sagemakerdp') and hyperparameters['sagemakerdp']:
        train_job_name = 'smdp-dist'
        distribution["smdistributed"]={ 
                            "dataparallel": {
                                "enabled": True
                            }
                    }

    else:
        distribution["mpi"]={"enabled": True}

    if do_spot_training:
        max_wait = max_run

    
    secret=get_secret()
    
    ## 
    code_commit_repo = f'https://git-codecommit.${region}.amazonaws.com/v1/repos/${git_repo_name}'  ### <== 5. source codecommit repository
    
    git_config = {'repo': code_commit_repo,
                  'branch': 'main',
                  'username': secret['username'],
                  'password': secret['password']}
    
    source_dir = 'yolov5'
    
    estimator = PyTorch(
        entry_point='train.py',
        source_dir=source_dir,
        git_config=git_config,
        role=role,
        sagemaker_session=sagemaker_session,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=1024,
        code_location = code_location,
        output_path=output_path,
        hyperparameters=hyperparameters,
        distribution=distribution,
        disable_profiler=True,
        debugger_hook_config=False,
        max_run=max_run,
        use_spot_instances=do_spot_training,
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
    )
    
    estimator.fit(
        inputs={'yolov5_input': s3_data_path},
        job_name=job_name,
        wait=False,
    )
    
    event['training_job_name'] = job_name
    event['stage'] = 'Training'
    
    return event
 

def get_secret():

    secret_name = "arn:aws:secretsmanager:${region}:${account-id}:secret:${secret-manager-name}"  ### <== 6. Secret Manager ARN 정보
    
    region_name = "ap-northeast-2"                                                                ### <== 7. region 명

    secret = {}
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
        
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
        secret = json.loads(secret)
    else:
        print("secret is not defined. Checking the Secrets Manager")

    return secret

