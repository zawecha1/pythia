# Copyright (c) Facebook, Inc. and its affiliates.
import glob
import importlib
import os
import sys
sys.path.append(r'C:/Users/zhangwc6/poc/pythia/')

from pythia.common.registry import registry
from pythia.utils.build_utils import build_trainer
from pythia.utils.distributed_utils import is_main_process
from pythia.utils.flags import flags

from _collections import defaultdict


class BaseVocab:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(
        self, vocab_file=None, embedding_dim=300, data_root_dir=None, *args, **kwargs
    ):
        """Vocab class to be used when you want to train word embeddings from
        scratch based on a custom vocab. This will initialize the random
        vectors for the vocabulary you pass. Get the vectors using
        `get_vectors` function. This will also create random embeddings for
        some predefined words like PAD - <pad>, SOS - <s>, EOS - </s>,
        UNK - <unk>.

        Parameters
        ----------
        vocab_file : str
            Path of the vocabulary file containing one word per line
        embedding_dim : int
            Size of the embedding

        """
        self.type = "base"
        self.word_dict = {}
        self.itos = {}

        self.itos[self.PAD_INDEX] = self.PAD_TOKEN
        self.itos[self.SOS_INDEX] = self.SOS_TOKEN
        self.itos[self.EOS_INDEX] = self.EOS_TOKEN
        self.itos[self.UNK_INDEX] = self.UNK_TOKEN

        self.word_dict[self.SOS_TOKEN] = self.SOS_INDEX
        self.word_dict[self.EOS_TOKEN] = self.EOS_INDEX
        self.word_dict[self.PAD_TOKEN] = self.PAD_INDEX
        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX

        index = len(self.itos.keys())

        self.total_predefined = len(self.itos.keys())

        if vocab_file is not None:
            if not os.path.isabs(vocab_file) and data_root_dir is not None:
                vocab_file = os.path.join(data_root_dir, vocab_file)
            if not os.path.exists(vocab_file):
                raise RuntimeError("Vocab not found at " + vocab_file)

            with open(vocab_file, "r") as f:
                for line in f:
                    self.itos[index] = line.strip()
                    self.word_dict[line.strip()] = index
                    index += 1

        self.word_dict[self.UNK_TOKEN] = self.UNK_INDEX
        # Return unk index by default
        self.stoi = defaultdict(lambda: self.UNK_INDEX)
        self.stoi.update(self.word_dict)


    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def get_size(self):
        return len(self.itos)

    def get_pad_index(self):
        return self.PAD_INDEX

    def get_pad_token(self):
        return self.PAD_TOKEN

    def get_start_index(self):
        return self.SOS_INDEX

    def get_start_token(self):
        return self.SOS_TOKEN

    def get_end_index(self):
        return self.EOS_INDEX

    def get_end_token(self):
        return self.EOS_TOKEN

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def get_vectors(self):
        return getattr(self, "vectors", None)


try:
    vob=BaseVocab(vocab_file='data/vocabs/vocabulary_captioning_thresh5.txt')
    print("success load vob")
except:
    import traceback
    traceback.print_exc()
    vob=None
    print("error in load vob")
	
	

def setup_imports():
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("pythia_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_pythia_path = os.environ.get("PYTHIA_PATH")

        if environment_pythia_path is not None:
            root_folder = environment_pythia_path

        root_folder = os.path.join(root_folder, "pythia")
        registry.register("pythia_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    tasks_folder = os.path.join(root_folder, "tasks")
    tasks_pattern = os.path.join(tasks_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("pythia.common.meter")

    files = glob.glob(tasks_pattern, recursive=True) + \
            glob.glob(model_pattern, recursive=True) + \
            glob.glob(trainer_pattern, recursive=True)

    for f in files:
        if f.endswith("task.py"):
            splits = f.split(os.sep)
            task_name = splits[-2]
            if task_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.tasks." + task_name + "." + module_name)
        elif f.find("models") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.models." + module_name)
        elif f.find("trainer") != -1:
            splits = f.split(os.sep)
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module("pythia.trainers." + module_name)
        elif f.endswith("builder.py"):
            splits = f.split(os.sep)
            task_name = splits[-3]
            dataset_name = splits[-2]
            if task_name == "tasks" or dataset_name == "tasks":
                continue
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            importlib.import_module(
                "pythia.tasks." + task_name + "." + dataset_name + "." + module_name
            )


def run():
    global vob
    setup_imports()
    parser = flags.get_parser()
    args = parser.parse_args()
    trainer = build_trainer(args)

    # Log any errors that occur to log file
    try:
        print("trainer", trainer)
        trainer.load()
        res = trainer.train()
        cap=res['captions'].cpu().numpy()[0]
        print("tools run res['captions']", cap)
        
        for i in cap[1:-1]:
            print(vob.itos[int(i)] + " ", end='')
        print('finish')
    except Exception as e:
        import traceback
        traceback.print_exc()
        writer = getattr(trainer, "writer", None)

        if writer is not None:
            writer.write(e, "error", donot_print=True)
        if is_main_process():
            raise


if __name__ == "__main__":
    import tensorflow as tf
    with tf.device('/cpu:0'):
        run()
        quit()