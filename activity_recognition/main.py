import gin
import logging
import wandb
import os
from absl import app, flags

from train import Trainer
# from diabetic_retinopathy.evaluation.eval import evaluate
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.modelLSTM import ModelRNN
# from models.modelLSTM import multi_rnn
from visualization.vis_guided import visualize
import tensorflow as tf

# to do:
# make window size variable (250 should not be fixed anywhere!!, produce multiple tfrecord files)
# make other model architectures applicable (Huibin)
# perform tuning

# include transition states (more classes)
# look in script what else we could implement?!
# PEP8 - Style Guide
# cleanup IDRID project + PEP8
# write ReadMe Files for both projects
# requirements.txt

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('visualize', False, 'Specify whether to show visualization or not')

# Define Model Architecture (Caution: only one of the following should be "True")
flags.DEFINE_boolean('conv_net', True, 'Specify whether to use vgg architecture')
flags.DEFINE_boolean('simple_lstm', False, 'Specify whether to use a very simple LSTM architecture')
flags.DEFINE_boolean('multi_rnn', False, 'Specify whether to use a multi_rnn architecture')


def main(argv):
    wandb.init()    ## TURN BACK ON WHEN TRAINING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    if FLAGS.conv_net:
        model = vgg_like(input_shape=(16, 250, 6), n_classes=6)
    elif FLAGS.simple_lstm:
        models = ModelRNN()
        model = models.my_LSTM()
    elif FLAGS.multi_rnn:
        models = ModelRNN()
        model = models.multi_rnn()
    else:
        print('ERROR: Specify a model architecture!')

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info,
                          run_paths)
        for _ in trainer.train():
            continue
    else:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        evaluate(model=model,
                 checkpoint=trainer.ckpt,
                 manager=trainer.manager,
                 ds_test=ds_test,
                 ds_info=ds_info,
                 run_paths=run_paths)

    if FLAGS.visualize:
        visualize(model=model, checkpoint=trainer.ckpt, manager=trainer.manager)
    wandb.finish()

if __name__ == "__main__":
    app.run(main)
