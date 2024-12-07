# https://refactoring.guru/design-patterns/singleton/python/example
import argparse
import os

import torch


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    arguments = {
        'b': {
            'name': 'benchmark',
            'default': "esbm_plus",
            'choices': ["esbm", "esbm_plus"],
            'help': "Benchmark version"
        },
        'd': {
            'name': 'dataset',
            'default': "dbpedia",
            'choices': ["dbpedia", "lmdb"],
            'help': "Target dataset"
        },
        'f': {
            'name': 'format',
            'default': "rdf",
            'choices': ["rdf", "json"],
            'help': "Data format"
        },
        's': {
            'name': 'seed',
            'default': -1,
            'type': int,
            'help': "Random seed"
        },
        'p': {
            'name': 'data_path',
            'default': "./data",
            'help': "Data path"
        },
        'O': {
            "name": "output_path",
            "default": "./output",
            "help": "Output path"
        },
        'tf': {
            "name": "target_feature",
            "default": "transe_nodes",
            "choices": ["transe_nodes", "transe_node_relations", "node_freq", "node_relation_freq"],
            "help": "Target feature to be used in the model"
        },
        'e': {
            "name": "encoder",
            "default": "RGCN",
            "choices": ["RGCN", "GCN"],
            "help": "Encoder type"
        },
        'o': {
            "name": "optimizer",
            "default": "Adam",
            "choices": ["ASGD", "AdamW", "Adam"],
            "help": "Optimizer type"
        },
        'sch': {
            "name": "scheduler",
            "default": "StepLR",
            "choices": ["StepLR", "OneCycleLR", "LinearLR", "ReduceLROnPlateau", "CosineAnnealingLR",
                        "CosineAnnealingWarmRestarts"],
            "help": "Scheduler type"
        },
        'lr': {
            "name": "learning_rate",
            "default": 0.01,
            "type": float,
            "help": "Learning rate"
        },
        'wd': {
            "name": "weight_decay",
            "default": 0.01,
            "type": float,
            "help": "Weight decay"
        },
        'ss': {
            "name": "step_size",
            "default":7,
            "type": int,
            "help": "Step size"
        },
        'ne': {
            "name": "num_epochs",
            "default": 50,
            "type": int,
            "help": "Number of epochs"
        },
        'nc': {
            "name": "num_cnn_layers",
            "default": 2,
            "type": int,
            "help": "Number of GCN layers"
        },
        'he': {
            "name": "hidden_embeddings",
            "default": 64,
            "type": int,
            "help": "Hidden embeddings size"
        },
        'af': {
            "name": "activation_function",
            "default": "ELU",
            "choices": ["ELU", "RReLU", "ReLU", "LeakyReLU"],
            "help": "Activation function"
        },
        'sl': {
            "name": "skip_layer",
            "default": False,
            "type": bool,
            "help": "Activate skip layer normalization"
        },
        'bn': {
            "name": "batch_norm",
            "default": True,
            "type": bool,
            "help": "Activate batch normalization"
        },
        'do': {
            "name": "drop_out",
            "default": 0.25,
            "type": float,
            "help": "Dropout "
        },
        'la': {
            "name": "loss_alpha",
            "default": 0.0,
            "type": float,
            "help": "Loss alpha factor"
        },
        'dt': {
            "name": "device_type",
            "default": "cpu",
            "choices": ["cuda", "cpu"],
            "help": "Device type"
        },
        'n': {
            "name": "num_executions",
            "default": 1,
            "type": int,
            "help": "Number times that model training will be executed to get the average performance"
        },
        'bm': {
            "name": "best_model",
            "default": True,
            "type": bool,
            "help": "Pick the best model based on the F1 score"
        }
    }

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        for k, v in self.arguments.items():
            choices = v['choices'] if 'choices' in v else None
            default = v['default'] if 'default' in v else None
            arg_type = v['type'] if 'type' in v else str
            if arg_type is bool:
                self.parser.add_argument(f"-{k}", f"--{v['name']}", type=lambda x: (str(x).lower() == 'true'), default=default,
                                     choices=choices, help=v['help'])
            else: 
                self.parser.add_argument(f"-{k}", f"--{v['name']}", type=arg_type, default=default,
                                     choices=choices, help=v['help'])
        self.args = self.parser.parse_args()

    def benchmark(self):
        return self.args.benchmark

    def dataset(self):
        return self.args.dataset

    def format(self):
        return self.args.format

    def seed(self):
        return self.args.seed

    def data_path(self):
        return self.args.data_path

    def output_path(self):
        os.makedirs(self.args.output_path, exist_ok=True)
        return self.args.output_path

    def target_feature(self):
        return self.args.target_feature

    def modularity_matrix_path(self):
        mmb = os.path.join(self.output_path(), 'Modularity_Matrix')
        os.makedirs(mmb, exist_ok=True)
        mmp = os.path.join(mmb, f'Modularity_{self.format()}_{self.dataset()}_{self.benchmark()}.npy')
        return mmp

    def encoder(self):
        return self.args.encoder

    def optimizer(self):
        return self.args.optimizer

    def scheduler(self):
        return self.args.scheduler

    def hidden_embeddings(self):
        return self.args.hidden_embeddings

    def learning_rate(self):
        return self.args.learning_rate

    def num_epochs(self):
        return self.args.num_epochs

    def weight_decay(self):
        return self.args.weight_decay

    def step_size(self):
        return self.args.step_size

    def num_cnn_layers(self):
        return self.args.num_cnn_layers

    def activation_function(self):
        if self.args.activation_function == "ELU":
            return torch.nn.ELU()
        elif self.args.activation_function == "ReLU":
            return torch.nn.ReLU()
        elif self.args.activation_function == "LeakyReLU":
            return torch.nn.LeakyReLU()
        return torch.nn.ELU()

    def skip_layer(self):
        return self.args.skip_layer

    def batch_norm(self):
        return self.args.batch_norm

    def drop_out(self):
        return self.args.drop_out

    def num_executions(self):
        return self.args.num_executions

    def loss_alpha(self):
        return self.args.loss_alpha

    def device(self):
        if self.args.device_type == "cuda" and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def best_model(self):
        return self.args.best_model

    def results_path(self):
        result_path = os.path.join(self.output_path(), 'results')
        os.makedirs(result_path, exist_ok=True)
        return result_path

    def model_path(self):
        model_path = os.path.join(self.output_path(), 'models', self.dataset(), self.benchmark())
        os.makedirs(model_path, exist_ok=True)
        return model_path

    def model_metadata(self, round_num):
        dataset_version = self.benchmark()
        dataset_name = self.dataset()
        dataset_type = self.format()
        weight_decay = self.weight_decay()
        learning_rate = self.learning_rate()
        hidden_embeddings = self.hidden_embeddings()
        feature_type = self.target_feature()
        encoder_type = self.encoder()
        optimizer_type = self.optimizer()
        return f'model_{dataset_type}_{dataset_name}_{dataset_version}_round_{round_num}_weight_decay_{weight_decay}_learning_rate_{learning_rate}_hidden_dim_{hidden_embeddings}_feature_type_{feature_type}_encoder_type{encoder_type}_optimizer_type_{optimizer_type}_model_condensed.pth'

    def result_metadata(self, round_num):
        dataset_version = self.benchmark()
        dataset_name = self.dataset()
        dataset_type = self.format()
        weight_decay = self.weight_decay()
        learning_rate = self.learning_rate()
        hidden_embeddings = self.hidden_embeddings()
        feature_type = self.target_feature()
        encoder_type = self.encoder()
        optimizer_type = self.optimizer()
        step_size = self.step_size()
        return f'results_{dataset_type}_{dataset_name}_{dataset_version}_weight_decay_{weight_decay}_learning_rate_{learning_rate}_hidden_dim_{hidden_embeddings}_feature_type_{feature_type}_encoder_type_{encoder_type}_optimizer_type_{optimizer_type}_step_size_{step_size}.csv'
