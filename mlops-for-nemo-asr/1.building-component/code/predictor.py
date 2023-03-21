
import os
import io
import copy
import json
import torch
import pathlib
import pickle
import sagemaker
import subprocess
import soundfile as sf
from tqdm.auto import tqdm

from nemo.collections.asr.models import EncDecCTCModel

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# cmd = ["df", "-h"]
# print(subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

def find_checkpoint(model_dir):
    checkpoint_path = None
    for (root, dirs, files) in os.walk(model_dir):
        if len(files) > 0:
            for file_name in files:
                if file_name.endswith('last.ckpt'):
                    checkpoint_path = root + '/' + file_name
    return checkpoint_path

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

    
def input_fn(request_body, input_content_type):  
    # print(f"***************** request_body : {request_body} ********************")
    print("***************** 1input_fn ********************")
    input_data = '/tmp'
    pathlib.Path(input_data).mkdir(parents=True, exist_ok=True)
    
    sf_data, samplerate = sf.read(io.BytesIO(request_body))
    filename = input_data + "/test.wav"
    sf.write(file=filename, data=sf_data, samplerate=samplerate)
    return filename

    

def model_fn(model_dir):
    print("***************** model_fn ********************")
    checkpoint_path = find_checkpoint(model_dir)

    asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    asr_model = asr_model.to(device)
    asr_model.eval()
    return asr_model


def predict_fn(filename, asr_model):
    print(f"***************** predict_fn ********************")
    result = asr_model.transcribe(paths2audio_files=[filename], batch_size=1)
    os.remove(filename)
    prediction = {"result" : result}
    return prediction
