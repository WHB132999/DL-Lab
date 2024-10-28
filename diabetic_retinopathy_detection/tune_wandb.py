import logging
import gin
import wandb
import math

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc

wandb.login()
# wandb.init(project="my-test-project", entity="sabruck")

def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder(','.join(bindings))

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        # gin.parse_config_files_and_bindings(['C:/Users/Samuel/PycharmProjects/diabetic/diabetic_retinopathy/configs/config.gin'],
        #                                     bindings) # change path to absolute path of config file
        gin.parse_config_files_and_bindings(
            ['/misc/home/RUS_CIP/st156810/dl-lab-21w-team15/diabetic_retinopathy/configs/config.gin'],
            bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load()

        # model
        # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
        model = vgg_like(input_shape=ds_info["image_shape"], n_classes=ds_info["n_classes"])

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue


sweep_config = {
    'name': 'diabetic_retinopathy_example',
    'method': 'random',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'Trainer.total_steps':
            {'distribution': 'q_uniform',
             'q': 1000,
             'min': 9000,
             'max': 12000
             },
        'vgg_like.base_filters': {
            'values': [10, 25, 50, 100]
            # 'distribution': 'q_log_uniform',
            # 'q': 1,
            # 'min': math.log(8),
            # 'max': math.log(128)
        },
        'vgg_like.n_blocks': {
            'values': [3, 4, 5]
            # 'distribution': 'q_uniform',
            # 'q': 1,
            # 'min': 2,
            # 'max': 6
        },
        'vgg_like.dense_units': {
            'values': [30, 50, 150, 170]
            # 'distribution': 'q_log_uniform',
            # 'q': 1,
            # 'min': math.log(16),
            # 'max': math.log(256)
        },
        'vgg_like.dropout_rate': {
            'values': [0.15, 0.2, 0.3, 0.55, 0.8]
            # 'distribution': 'uniform',
            # 'min': 0.1,
            # 'max': 0.9
        }
    }
}
sweep_id = wandb.sweep(sweep_config)

wandb.agent(sweep_id, function=train_func, count=30)