import os
import json
import glob
import tarfile
import argparse
import subprocess
from distutils.dir_util import copy_tree

class preprocess():
    
    def __init__(self, args):
        
        self.args = args
        self.proc_prefix = self.args.proc_prefix #'/opt/ml/processing'
        
        self.input_dir = os.path.join(self.proc_prefix, "input")
        self.output_dir = os.path.join(self.proc_prefix, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _sph_to_wav(self, data_dir):
        
        an4_path = os.path.join(data_dir, "an4_sphere.tar.gz")
        
        if not os.path.exists(data_dir + "/an4/"):
            tar = tarfile.open(an4_path)
            tar.extractall(path=data_dir)

            print("Converting .sph to .wav...")
            sph_list = glob.glob(data_dir + '/an4/**/*.sph', recursive=True)
            for sph_path in sph_list:
                wav_path = sph_path[:-4] + '.wav'
                cmd = ["sox", sph_path, wav_path]
                subprocess.run(cmd)
                
        print("Finished conversion.\n******")
        
    # Function to build a manifest
    def _build_manifest(self, transcripts_path, manifest_path, data_dir, mount_dir, wav_path):
        # create manifest with reference to this directory. This is useful when mounting the dataset.
        mount_dir = mount_dir if mount_dir else data_dir
        with open(transcripts_path, 'r') as fin:
            with open(manifest_path, 'w') as fout:
                for line in fin:
                    # Lines look like this:
                    # <s> transcript </s> (fileID)
                    transcript = line[: line.find('(') - 1].lower()
                    transcript = transcript.replace('<s>', '').replace('</s>', '')
                    transcript = transcript.strip()

                    file_id = line[line.find('(') + 1 : -2]  # e.g. "cen4-fash-b"
                    audio_path = os.path.join(
                        data_dir, wav_path, file_id[file_id.find('-') + 1 : file_id.rfind('-')], file_id + '.wav'
                    )

                    mounted_audio_path = os.path.join(
                        mount_dir, wav_path, file_id[file_id.find('-') + 1 : file_id.rfind('-')], file_id + '.wav'
                    )
                    # import sox here to not require sox to be available for importing all utils.
                    import sox

                    duration = sox.file_info.duration(audio_path)

                    # Write the metadata to the manifest
                    metadata = {"audio_filepath": mounted_audio_path, "duration": duration, "text": transcript}
                    json.dump(metadata, fout)
                    fout.write('\n')


    def _make_manifest(self, data_dir, train_mount_dir, test_mount_dir):
        
        # Building Manifests
        print("******")
        train_transcripts = data_dir + '/an4/etc/an4_train.transcription'
        train_manifest = data_dir + '/an4/train_manifest.json'

        if not os.path.isfile(train_manifest):
            self._build_manifest(train_transcripts, train_manifest, data_dir, train_mount_dir, 'an4/wav/an4_clstk')
            print("Training manifest created.")

        test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
        test_manifest = data_dir + '/an4/test_manifest.json'
        if not os.path.isfile(test_manifest):
            self._build_manifest(test_transcripts, test_manifest, data_dir, test_mount_dir, 'an4/wav/an4test_clstk')
            print("Test manifest created.")
        print("***Done***")
        
    def _delete_sph(self, data_dir):
        
        for (root, dirs, files) in os.walk(data_dir):
            if len(files) > 0:
                for file_name in files:
                    if file_name.endswith('.sph'):
                        os.remove(root + '/' + file_name)
                        
        os.remove(os.path.join(data_dir, "an4_sphere.tar.gz"))
        
    def execution(self, ):
        
        self._sph_to_wav(
            data_dir=self.input_dir
        )
        
        self._make_manifest(
            data_dir=self.input_dir,
            train_mount_dir=self.args.train_mount_dir,
            test_mount_dir=self.args.test_mount_dir
        )
        
        self._delete_sph(
            data_dir=self.input_dir
        )
        
        
        copy_tree(os.path.join(self.input_dir, "an4"), os.path.join(self.output_dir, "an4"))
        
        print ("data_dir", os.listdir(self.input_dir))
        print ("self.output_dir", os.listdir(self.output_dir))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_prefix", type=str, default="/opt/ml/processing")
    parser.add_argument("--train_mount_dir", type=str, default="train_mount_dir")
    parser.add_argument("--test_mount_dir", type=str, default="test_mount_dir")
    
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))

    prep = preprocess(args)
    prep.execution()