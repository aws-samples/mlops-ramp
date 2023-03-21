import os
import copy
import boto3
import logging
import json
import jsonlines # !pip install jsonlines 해주기
import torch
import tarfile

from tqdm.auto import tqdm
from omegaconf import open_dict
from nemo.collections.asr.models import EncDecCTCModel

# import glob
import pickle
import sox
import time
import io
import soundfile as sf
import base64
import numpy as np
import pathlib
from sagemaker.s3 import S3Downloader
from datetime import datetime

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.transcribe_utils import PunctuationCapitalization
from nemo.core.config import hydra_runner
from nemo.utils import logging


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


def find_checkpoint(model_dir):
    checkpoint_path = None
    for (root, dirs, files) in os.walk(model_dir):
        if len(files) > 0:
            for file_name in files:
                if file_name.endswith('last.ckpt'):
                    checkpoint_path = root + '/' + file_name
    return checkpoint_path


def find_files(jsonl_dir):
    jsonl_list = []
    for (root, dirs, files) in os.walk(jsonl_dir):
        if len(files) > 0:
            for file_name in files:
                if file_name.endswith('jsonl'):
                    jsonl_list.append(root + '/' + file_name)
    return jsonl_list


def read_manifest(path):
    manifest = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc="Reading manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            manifest.append(data)
    return manifest


def write_processed_manifest(data, original_path):
    original_manifest_name = os.path.basename(original_path)
    new_manifest_name = original_manifest_name.replace(".json", "_processed.json")

    manifest_dir = os.path.split(original_path)[0]
    filepath = os.path.join(manifest_dir, new_manifest_name)
    with open(filepath, 'w') as f:
        for datum in tqdm(data, desc="Writing manifest data"):
            datum = json.dumps(datum)
            f.write(f"{datum}\n")
    print(f"Finished writing manifest: {filepath}")
    return filepath


def apply_preprocessors(manifest, preprocessors):
    for processor in preprocessors:
        for idx in tqdm(range(len(manifest)), desc=f"Applying {processor.__name__}"):
            manifest[idx] = processor(manifest[idx])

    print("Finished processing manifest !")
    return manifest


def change_dir(data):
    MANIFEST_PATH = os.environ['MANIFEST_PATH']
    WAV_PATH = os.environ['WAV_PATH']
    data['audio_filepath'] = data['audio_filepath'].replace(MANIFEST_PATH, WAV_PATH)
    return data


def predict(asr_model, predictions, targets, target_lengths, predictions_lengths=None):
    references = []
    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = targets.long().cpu()
        tgt_lenths_cpu_tensor = target_lengths.long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = asr_model.decoding.decode_tokens_to_str(target)
            references.append(reference)

        hypotheses, _ = asr_model.decoding.ctc_decoder_predictions_tensor(
            predictions, predictions_lengths, fold_consecutive=True
        )
    return references[0], hypotheses[0]

def start_retraining_codepipeline():
    region_name = os.environ["region"]
    sm_client = boto3.client('sagemaker', region_name=region_name)
    pipeline_client = boto3.client('codepipeline', region_name=region_name)

    response = sm_client.list_projects(
      SortBy='CreationTime',
      SortOrder='Descending'
    )

    for pjt_list in response['ProjectSummaryList']:
        if pjt_list['ProjectStatus'] == 'CreateCompleted':
            ProjectName = pjt_list['ProjectName']
            break

    des_response = sm_client.describe_project(
        ProjectName=ProjectName
    )

    code_pipeline_name = f"sagemaker-{des_response['ProjectName']}-{des_response['ProjectId']}-modelbuild"
    pipeline_client.start_pipeline_execution(name=code_pipeline_name) 
    print("Start retraining ........")
    
def main():
    
    reference_list = []
    predicted_list = []    
        
    if not os.environ.get("select_date"):  

        
        model_path = "/opt/ml/processing/model/model.tar.gz"
        model_dir = 'trained_model'
        with tarfile.open(model_path) as tar:
            tar.extractall(path=model_dir)

        print("Loading nemo model.")
        checkpoint_path = find_checkpoint(model_dir)
        print(f"checkpoint_path : {checkpoint_path}")

        asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
        asr_model = asr_model.to(device)
        asr_model.eval()

        print("Reading test data.")    

        test_manifest_filename = "/opt/ml/processing/input/manifest/test_manifest.json"
        eval_test_manifest_filename = "/opt/ml/processing/evaluation/eval_test_manifest.json"

        test_data = read_manifest(test_manifest_filename)
        test_data_processed = apply_preprocessors(test_data, [change_dir])
        local_test_manifest_path = write_processed_manifest(test_data_processed, eval_test_manifest_filename)

        cfg = copy.deepcopy(asr_model.cfg)

        with open_dict(cfg):
            cfg.test_ds.manifest_filepath = local_test_manifest_path

        asr_model.setup_multiple_test_data(cfg.test_ds)

        wer_nums = []
        wer_denoms = []


        for test_batch in asr_model.test_dataloader():
            test_batch = [x.cuda() for x in test_batch]
            targets = test_batch[2]
            targets_lengths = test_batch[3]

            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
            # Notice the model has a helper object to compute WER
            asr_model._wer.update(greedy_predictions, targets, targets_lengths)
            reference, predicted = predict(asr_model, greedy_predictions, targets, targets_lengths)

            print(f"*********** reference : {reference}")
            print(f"*********** predicted : {predicted}")
            reference_list.append(reference)
            predicted_list.append(predicted)

            _, wer_num, wer_denom = asr_model._wer.compute()
            asr_model._wer.reset()
            wer_nums.append(wer_num.detach().cpu().numpy())
            wer_denoms.append(wer_denom.detach().cpu().numpy())

            # Release tensors from GPU memory
            del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions
        # We need to sum all numerators and denominators first. Then divide.
        wer_result = sum(wer_nums)/sum(wer_denoms)

    else:
                
        select_date = os.environ["select_date"]
        
        # output_list = S3Downloader.list(inference_output_s3uri + f'/output_monitor/{endpoint_name}/{target_model}/{select_date}')
        output_list = find_files('/opt/ml/processing/input/inference_data')
        print(f"output_list: {output_list}")
        with open('/opt/ml/processing/input/manifest/gt_manifest.pkl', 'rb') as f:
            gt_list = pickle.load(f)
            
            
        result_data = []

        train_mount_dir=f"/opt/ml/input/data/training/"
        test_mount_dir=f"/opt/ml/input/data/testing/"
        manifest_path = f"/opt/ml/processing/output/{select_date}/manifest"
        manifest_file = f"{manifest_path}/test_manifest.json"
        result_wav_file = f"/opt/ml/processing/output/{select_date}/wav"
        
        pathlib.Path(manifest_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(result_wav_file).mkdir(parents=True, exist_ok=True)
        
        seq = 0
        with open(manifest_file, 'w') as fout:
            for json_list in output_list:
                print(json_list)
                # Read a specific file
                
                fname = json_list.split('/')[-1]
                fname = fname.split('.')[0]
                f_date = select_date.replace('/','-')
                with jsonlines.open(json_list) as read_file:
                    for res in read_file.iter():
                # for line in json_data.splitlines():
                #     res = json.loads(line)
                    # print(res)
                        filename = f"{result_wav_file}/{f_date}-{fname}-{seq}.wav"
                        sf_data, samplerate = sf.read(io.BytesIO(base64.b64decode(res['captureData']['endpointInput']['data'])))
                        sf.write(file=filename, data=sf_data, samplerate=samplerate)
                        # print(base64.b64decode(res['captureData']['endpointOutput']['data']))
                        np_val = np.load(io.BytesIO(base64.b64decode(res['captureData']['endpointOutput']['data'])), allow_pickle=True)
                        transcript = ' '.join(np_val.item()['result'])
                        predicted_list.append(transcript)
                        reference_list.append(gt_list[seq])


                        mounted_audio_path = filename.replace(result_wav_file, test_mount_dir)

                        # import sox here to not require sox to be available for importing all utils.
                        duration = sox.file_info.duration(filename)

                        # Write the metadata to the manifest
                        metadata = {"audio_filepath": mounted_audio_path, "duration": duration, "pred_text": transcript}
                        json.dump(metadata, fout)
                        fout.write('\n')
                        seq += 1
                    
        pc = PunctuationCapitalization('.,?')
        reference_list = pc.separate_punctuation(reference_list)
        reference_list = pc.do_lowercase(reference_list)
        predicted_list = pc.do_lowercase(predicted_list)
        reference_list = pc.rm_punctuation(reference_list)
        predicted_list = pc.rm_punctuation(predicted_list)


        # Compute the WER
        cer = word_error_rate(hypotheses=predicted_list, references=reference_list, use_cer=True)
        wer = word_error_rate(hypotheses=predicted_list, references=reference_list, use_cer=False)
        
        use_cer = False
        
        if use_cer:
            metric_name = 'CER'
            metric_value = cer
        else:
            metric_name = 'WER'
            metric_value = wer
        
        tolerance = float(os.environ['tolerance'])
        
        print(f" tolerance : {tolerance}")
        print(f" tolerance : {type(tolerance)}")
        print(f" metric_value : {metric_value}")
        print(f" metric_value : {type(metric_value)}")
        
        
        if tolerance is not None:
            if metric_value > tolerance:
                print(f"Got {metric_name} of {metric_value}, which was higher than tolerance={tolerance}")
                start_retraining_codepipeline()
                
            print(f'Got {metric_name} of {metric_value}. Tolerance was {tolerance}')
        else:
            print(f'Got {metric_name} of {metric_value}')

        print(f'Dataset WER/CER ' + str(round(100 * wer, 2)) + "%/" + str(round(100 * cer, 2)) + "%")
    
        wer_result = wer

    
    report_dict = {
        "metrics": {
            "wer": {
                "value": wer_result
            },
        },
        "reference": {
            "value": reference_list
        },
        "predicted": {
            "value": predicted_list
        }
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Writing out evaluation report with wer: %f", wer_result)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))


if __name__ == '__main__':
    main()