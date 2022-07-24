"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

# Additional dependencies
from sagemaker.processing import FrameworkProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker import image_uris

from pipelines.fraud.ag_model import (
    AutoGluonTraining,
    AutoGluonInferenceModel,
    AutoGluonTabularPredictor,
    AutoGluonFramework
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print(BASE_DIR)

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="FraiudPackageGroup",
    pipeline_name="FraudPipeline",
    base_job_prefix="Fraud",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{default_bucket}/autogluon/dataset",
    )
    config_data = ParameterString(
        name="ConfigDataUrl",
        default_value=f"s3://{default_bucket}/autogluon/config",
    )

#     # processing step for feature engineering
#     sklearn_processor = SKLearnProcessor(
#         framework_version="0.23-1",
#         instance_type=processing_instance_type,
#         instance_count=processing_instance_count,
#         base_job_name=f"{base_job_prefix}/sklearn-abalone-preprocess",
#         sagemaker_session=pipeline_session,
#         role=role,
#     )
#     step_args = sklearn_processor.run(
#         outputs=[
#             ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
#             ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
#             ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
#         ],
#         code=os.path.join(BASE_DIR, "preprocess.py"),
#         arguments=["--input-data", input_data],
#     )
#     step_process = ProcessingStep(
#         name="PreprocessAbaloneData",
#         step_args=step_args,
#     )

    # training step for generating model artifacts
    hyperparameters = {
           "config_name" : "config-med.yaml"
    }

    max_run = 1*60*60

    use_spot_instances = False
    if use_spot_instances:
        max_wait = 1*60*60
    else:
        max_wait = None

    base_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/FarudTrain"

    ag_estimator = AutoGluonTraining(
        entry_point="autogluon_starter_script.py",
        source_dir=f"{BASE_DIR}/src",
        role=role,
        # region=region,
        sagemaker_session=pipeline_session,
        output_path=base_path,
        code_location=base_path,
        hyperparameters=hyperparameters,
        instance_count=1,
        instance_type=training_instance_type,
        framework_version="0.4",
        py_version="py38",
        max_run=max_run,
        use_spot_instances=use_spot_instances,  # spot instance 활용
        max_wait=max_wait,
        base_job_name=f"fraud-train", # base_job_name 적용
    #     disable_profiler=True
    )

    data_channels = {
        "inputdata": TrainingInput(s3_data=input_data),
        "config": TrainingInput(s3_data=config_data)
    }
    
    step_train = TrainingStep(
        name="TrainFraudModel",
        estimator=ag_estimator,
        inputs=data_channels
    #     cache_config=CacheConfig(enable_caching=True, expire_after='T12H')
    )

    # processing step for evaluation
    image_uri = image_uris.retrieve(
        "autogluon",
        region=region,
        version="0.4",
        py_version="py38",
        image_scope="training",
        instance_type=processing_instance_type,
    )
    
    script_eval = FrameworkProcessor(
        AutoGluonFramework,
        framework_version="0.4",
        role=role,
        py_version="py38",
        image_uri=image_uri,
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/FraudEval/fraud-eval",
        sagemaker_session=pipeline_session,
    )
    

    step_args = script_eval.run(
        code="autogluon_evaluation.py",
        source_dir=f"{BASE_DIR}/src",
        inputs=[
            ProcessingInput(
                source=input_data, 
                input_name="test_data", 
                destination="/opt/ml/processing/test"
            ),
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts, 
                input_name="model_weight", 
                destination="/opt/ml/processing/model"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output", 
                output_name='evaluation'),
        ],
        wait=False
    )

    evaluation_report = PropertyFile(
        name="FraudEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateFraudModel",
        step_args=step_args,
        property_files=[evaluation_report],
    )    


    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    model_image_uri = image_uris.retrieve(
        "autogluon",
        region=region,
        version="0.4",
        py_version="py38",
        image_scope="inference",
        instance_type="ml.m5.large",
    )
    
    base_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/FraudModel"
    model = AutoGluonInferenceModel(
        source_dir=f"{BASE_DIR}/src",
        entry_point="autogluon_serve.py",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        instance_type="ml.m5.large",
        role=role,
        sagemaker_session=pipeline_session,
        framework_version="0.4",
        py_version="py38",
        predictor_cls=AutoGluonTabularPredictor
    )
    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    step_register = ModelStep(
        name="RegisterFraudModel",
        step_args=step_args,
    )

    # condition step for evaluating model quality and branching execution
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.accuracy.value"
        ),
        right=0.9,
    )
    step_cond = ConditionStep(
        name="CheckAccuracyFraudEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            training_instance_type,
            model_approval_status,
            input_data,
            config_data
        ],
        steps=[step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
