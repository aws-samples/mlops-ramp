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
    FrameworkProcessor,
)
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
    CacheConfig,
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from pipelines.nemo_asr.config.config import config_handler
from utils.ssm import parameter_store
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables

from time import strftime
from smexperiments.trial import Trial
from smexperiments.experiment import Experiment
from sagemaker.workflow.retry import StepRetryPolicy, StepExceptionTypeEnum, SageMakerJobExceptionTypeEnum, SageMakerJobStepRetryPolicy
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.pytorch.model import PyTorchModel
from omegaconf import OmegaConf
import shutil


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    
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
            ResourceArn=sagemaker_project_arn.lower())
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

class sm_pipeline():
    
    def __init__(self, region, sagemaker_project_arn, role, default_bucket, \
                 pipeline_name, model_package_group_name, base_job_prefix, \
                ):
        
        #self.pipeline_config = config_handler(strConfigPath="config-pipeline.ini")
        
        self.region=region
        self.default_bucket = default_bucket
        self.model_package_group_name = model_package_group_name
        self.pm = parameter_store(self.region)
        if role is None: self.role = sagemaker.session.get_execution_role()
        else: self.role=role
        if base_job_prefix is None: self.base_job_prefix = self.pm.get_params(key="PREFIX") #self.pipeline_config.get_value("COMMON", "base_job_prefix")
        else: self.base_job_prefix = base_job_prefix
        self.prefix = self.pm.get_params(key="PREFIX")
        if pipeline_name is None: self.pipeline_name = self.base_job_prefix + "-pipeline" #self.pipeline_config.get_value("PIPELINE", "name")
        else: self.pipeline_name = pipeline_name

        self.sagemaker_session = get_session(self.region, default_bucket)
        self.pipeline_session = get_pipeline_session(region, default_bucket)
        
        self.cache_config = CacheConfig(
            enable_caching=True, #self.pipeline_config.get_value("PIPELINE", "enable_caching", dtype="boolean"),
            expire_after="T48H" #self.pipeline_config.get_value("PIPELINE", "expire_after")
        )    
        
        self.git_config = {
            'repo': f'https://{self.pm.get_params(key="-".join([self.prefix, "CODE_REPO"]))}', ### <== 5. git repository 위치 수정
            'branch': 'main',
            'username': self.pm.get_params(key="-".join([self.prefix, "CODECOMMIT-USERNAME"]), enc=True),
            'password': self.pm.get_params(key="-".join([self.prefix, "CODECOMMIT-PWD"]), enc=True)
        }
        
        self.retry_policies=[                
            # retry when resource limit quota gets exceeded
            SageMakerJobStepRetryPolicy(
                exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
                expire_after_mins=180,
                interval_seconds=60,
                backoff_rate=1.0
            ),
        ]
        
        #self.pipeline_config.set_value("PREPROCESSING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        #self.pipeline_config.set_value("TRAINING", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        #self.pipeline_config.set_value("EVALUATION", "image_uri", self.pm.get_params(key=''.join([self.base_job_prefix, "IMAGE-URI"])))
        #pm.get_params(key="-".join([prefix, "IMAGE-URI"]))
        
        #self.model_approval_status = self.pipeline_config.get_value("MODEL_REGISTER", "model_approval_status_default") #"PendingManualApproval"
        
        
        # self.model_approval_status = ParameterString(
        #     name="ModelApprovalStatus",
        #     default_value="PendingManualApproval"
        # )
        
        self.proc_prefix = "/opt/ml/processing"        
        self.input_data_path = self.pm.get_params(key="-".join([self.prefix, "S3-DATA-PATH"])) #self.pipeline_config.get_value("INPUT", "input_data_s3_uri") 
        
    def create_trial(self, experiment_name):
        
        create_date = strftime("%m%d-%H%M%s")
        sm_trial = Trial.create(trial_name=f'{experiment_name}-{create_date}',
                                experiment_name=experiment_name)
        job_name = f'{sm_trial.trial_name}'
        return job_name
    
    def create_experiment(self, experiment_name):
        try:
            sm_experiment = Experiment.load(experiment_name)
        except:
            sm_experiment = Experiment.create(experiment_name=experiment_name)
            
        return sm_experiment
    
    def _find_nemo_ckpt(self, model_dir):
        checkpoint_path = None
        for (root, dirs, files) in os.walk(model_dir):
            if len(files) > 0:
                for file_name in files:
                    if file_name.endswith('.nemo'):
                        checkpoint_path = root + '/' + file_name
        return checkpoint_path

    def _get_model_data(self, ):
        
        sm_client = boto3.client('sagemaker')
        response = sm_client.list_model_packages(ModelPackageGroupName=self.model_package_group_name)
        
        model_arn = None
        for model in response['ModelPackageSummaryList']:
            approval_status = model["ModelApprovalStatus"]
            if approval_status == "Approved":
                model_arn = model['ModelPackageArn']
                break
        
        if model_arn is None: return None
        else:
            
            import tarfile
            from utils.s3 import s3_handler
            
            s3 = s3_handler()
            
            response = sm_client.describe_model_package(ModelPackageName=model_arn)
            model_data_url = response['InferenceSpecification']["Containers"][0]["ModelDataUrl"]
            #return model_data_url
            
            print ("model_data_url", model_data_url)
            
            os.makedirs("./tmp", exist_ok=True)
            s3.download_obj(
                source_bucket=model_data_url.split("/", 3)[2],
                source_obj=model_data_url.split("/", 3)[3],
                target_file="./tmp/model.tar.gz"
            )
            
            model_path = "./tmp/model.tar.gz"
            model_dir = './tmp/trained_model'
            with tarfile.open(model_path) as tar:
                tar.extractall(path=model_dir)

            print("Loading nemo model.")
            checkpoint_path = self._find_nemo_ckpt(model_dir)
            print(f"checkpoint_path : {checkpoint_path}")
            
            pretrained_model_s3_path = s3.upload_file(
                source_file=checkpoint_path,
                target_bucket=self.default_bucket,
                target_obj=os.path.join(
                    self.pipeline_name,
                    "training",
                    "pre-trained",
                    os.path.basename(checkpoint_path)
                )
            )
            if os.path.isdir("./tmp"): shutil.rmtree("./tmp")            
            print ("pretrained_model_s3_path", pretrained_model_s3_path)
            return pretrained_model_s3_path
    
    def _step_preprocessing(self, ):
        


        '''
            1. processor for preprocessing task 정의
              : 1.nemo-preprocessing-job 참조
          
        '''
        ################
        ## your codes ##
        ################
        
        '''
            2. step argument 정의
              : 1.nemo-preprocessing-job 참조
          
        '''
        
        step_args = dataset_processor.run(
            job_name="preprocessing", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            
            ################
            ## your codes ##
            ################
        )
        
        self.preprocessing_process = ProcessingStep(
            name="PreprocessingProcess", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
            retry_policies=self.retry_policies
        )
        
        print ("  \n== Preprocessing Step ==")
        print ("   \nArgs: ", self.preprocessing_process.arguments.items())
        
    def _step_preprocessing_2(self, ):
        
        '''
            1. processor for preprocessing task 정의
              : 1.nemo-preprocessing-job 참조
          
        '''
        ################
        ## your codes ##
        ################
        
        '''
            2. step argument 정의
              : 1.nemo-preprocessing-job 참조
          
        '''
        
        step_args = dataset_processor.run(
            job_name="preprocessing-2", ## 이걸 넣어야 캐시가 작동함, 안그러면 프로세서의 base_job_name 이름뒤에 날짜 시간이 붙어서 캐시 동작 안함
            
            ################
            ## your codes ##
            ################
        )
        
        self.preprocessing_process_2 = ProcessingStep(
            name="PreprocessingProcess-2", ## Processing job이름
            step_args=step_args,
            cache_config=self.cache_config,
            retry_policies=self.retry_policies
        )
        
        print ("  \n== Preprocessing Step 2 ==")
        print ("   \nArgs: ", self.preprocessing_process_2.arguments.items())
    
    

    def _step_training(self, ):
        
        
        config_path = os.path.join("./code", "conf", "config.yaml")
        ## config modification for retraining
        if self.pm.get_params(key="-".join([self.prefix, "RETRAIN"])) == "True":
            
            pretrained_model_data_url = self._get_model_data()
            
            if pretrained_model_data_url is None: print ("No approved model")
            else:
                conf = OmegaConf.load(config_path)
                # resume flags if crashes occur
                conf.exp_manager.resume_if_exists=True
                conf.exp_manager.resume_ignore_no_checkpoint=True
                # the pre-trained model we want to fine-tune
                conf.init_from_nemo_model = f"/opt/ml/input/data/pretrained/{os.path.basename(pretrained_model_data_url)}" #"/opt/ml/input/data/pretrained/블라블라블라.nemo"
                OmegaConf.save(conf, config_path)
                
                pretrain_s3_path = pretrained_model_data_url
                print ("pretrained_model_data_url", pretrain_s3_path)
                self.pm.put_params(key="-".join([self.prefix, "RETRAIN"]), value=False, overwrite=True)
        else:
            
            print ("here2")
            
            conf = OmegaConf.load(config_path)
            conf.exp_manager.resume_if_exists=False 
            conf.exp_manager.resume_ignore_no_checkpoint=False
            conf.init_from_nemo_model = None
            OmegaConf.save(conf, config_path)

            pretrain_s3_path = self.pm.get_params(key=self.prefix + "-PRETRAINED-WEIGHT")
        
        print ("pretrain_s3_path", pretrain_s3_path)
        
        num_re = "([0-9\\.]+)(e-?[[01][0-9])?"
        
        '''
            1. estimator for training task 정의
              : 2.nemo-training-job 참조
          
        '''
        ################
        ## your codes ##
        ################
        
        sm_experiment = self.base_job_prefix + "train-exp" #self.create_experiment(self.base_job_prefix + self.pipeline_config.get_value("TRAINING", "experiment_name"))
        job_name = self.base_job_prefix + "train-exp" #self.create_trial(self.base_job_prefix + self.pipeline_config.get_value("TRAINING", "experiment_name"))
        
        '''
            2. step argument 정의
              : 2.nemo-training-job 참조
          
        '''
        
        step_training_args = self.estimator.fit(
            
            ################
            ## your codes ##
            ################
            
        )

        
        self.training_process = TrainingStep(
            name="TrainingProcess",
            step_args=step_training_args,
            cache_config=self.cache_config,
            depends_on=[self.preprocessing_process, self.preprocessing_process_2],
            retry_policies=self.retry_policies
        )
            
        print ("  \n== Training Step ==")
        print ("   \nArgs: ", self.training_process.arguments.items())
        
    def _step_evaluation(self, ):
        
        
        '''
            1. processor for evaluation task 정의
              : 3.nemo-evaluation-job 참조
          
        '''
        ################
        ## your codes ##
        ################
        
        sm_experiment = self.base_job_prefix + "eval-exp" #self.create_experiment(self.base_job_prefix + self.pipeline_config.get_value("EVALUATION", "experiment_name"))
        job_name = self.base_job_prefix + "eval-exp" #self.create_trial(self.base_job_prefix + self.pipeline_config.get_value("EVALUATION", "experiment_name"))
        
        self.evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation-metrics", ## evaluation의 ProcessingOutput의 output_name
            path="evaluation.json", ## evaluate.py 에서 write하는 부분
        )
        
        '''
            2. step argument 정의
              : 3.nemo-evaluation-job 참조
          
        '''
        
        step_args = eval_processor.run(
            
            ################
            ## your codes ##
            ################
           
        )
        
        self.evaluation_process = ProcessingStep(
            name="EvaluationProcess", ## Processing job이름
            step_args=step_args,
            # depends_on=[self.training_process],
            property_files=[self.evaluation_report],
            cache_config=self.cache_config,
            retry_policies=self.retry_policies
        )
        
        print ("  \n== Evaluation Step ==")
        print ("   \nArgs: ", self.evaluation_process.arguments.items())
        
    def _step_model_registration(self, ):
      
        #self.model_package_group_name = ''.join([self.prefix, self.model_name])
        self.pm.put_params(key=self.prefix + "MODEL-GROUP-NAME", value=self.model_package_group_name, overwrite=True)
                                                                              
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(
                    on="/",
                    values=[
                        self.evaluation_process.properties.ProcessingOutputConfig.Outputs["evaluation-metrics"].S3Output.S3Uri,
                        #print (self.evaluation_process.arguments.items())로 확인가능
                        "evaluation.json"
                    ],
                ),
                content_type="application/json"
            )
        )

        model = PyTorchModel(
            entry_point="predictor.py",
            source_dir="./code/",
            git_config=self.git_config,
            code_location=os.path.join(
                "s3://",
                self.default_bucket,
                self.pipeline_name,
                "inference",
                "model"
            ),
            model_data=self.training_process.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.role,
            image_uri=self.pm.get_params(key="-".join([self.prefix, "INF-IMAGE-URI"])),
            framework_version="1.13.1",
            sagemaker_session=self.pipeline_session,
        )
        
        step_args = model.register(
            content_types=["file-path/raw-bytes", "text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.g4dn.xlarge"],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name=self.model_package_group_name,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
            
        )
        self.register_process = ModelStep(
            name="ModelRegisterProcess",
            step_args=step_args,
            # depends_on=[self.evaluation_process]
        )
        
            # condition step for evaluating model quality and branching execution
        cond_lte = ConditionLessThanOrEqualTo(
            left=JsonGet(
                step_name=self.evaluation_process.name,
                property_file=self.evaluation_report,
                json_path="metrics.wer.value"
            ),
            right=50.0,
        )
        
        self.step_cond = ConditionStep(
            name="CheckWEREvaluation",
            conditions=[cond_lte],
            if_steps=[self.register_process],
            else_steps=[],
        )

        
        print ("  \n== Registration Step ==")

    def _execution_steps(self, ):
        
        self._step_preprocessing()
        self._step_preprocessing_2()
        self._step_training()
        self._step_evaluation()
        self._step_model_registration()
        
    def get_pipeline(self, ):
        
        self._execution_steps()
        
        pipeline = Pipeline(
            name=self.pipeline_name,
            steps=[self.preprocessing_process, self.preprocessing_process_2, \
                   self.training_process, \
                   self.evaluation_process, \
                   self.step_cond, \
            ],
            # steps=[self.preprocessing_process, self.preprocessing_process_2, \
            #        self.training_process, \
            # ],
            sagemaker_session=self.pipeline_session
        )

        return pipeline
        
def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    pipeline_name=None,
    model_package_group_name="NEMOASRPackageGroup",
    base_job_prefix="NEMOASR",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """

    nemo_asr_pipeline = sm_pipeline(
        region=region,
        sagemaker_project_arn=sagemaker_project_arn,
        role=role,
        default_bucket=default_bucket,
        pipeline_name=pipeline_name,
        model_package_group_name=model_package_group_name,
        base_job_prefix=base_job_prefix
    )
    
    return nemo_asr_pipeline.get_pipeline()