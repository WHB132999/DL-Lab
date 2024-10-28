import gin
import logging
import wandb
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models.Resnet_model import resnet18, resnet34
from models.Transfer_models import MyMobileNetV2, inception_v3, inception_resnet_v2
from visualization.vis_guided import visualize
from visualization.vis_guided_an import visualize_an

# CREATED BY SAMUEL BRUCKER AND HUIBIN WANG

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('visualize', True, 'Specify whether to make visualization images or not')
flags.DEFINE_boolean('visualize_an', True, 'Specify whether to make an_visualization images or not')

# VGG-Model
flags.DEFINE_boolean('vgg', True, 'Specify whether to use VGG-model or not')

# Resnet-Models
flags.DEFINE_boolean('resnet18', False, 'Specify whether to use Resnet18 or not')
flags.DEFINE_boolean('resnet34', False, 'Specify whether to use Resnet34 or not')

# Transfer-Models
flags.DEFINE_boolean('mobilenet_v2', False, 'Specify whether to use Transfer-model MobileNetV2 or not')
flags.DEFINE_boolean('inception_v3', False, 'Specify whether to use Transfer-model InceptionV3 or not')
flags.DEFINE_boolean('inception_resnet_v2', False, 'Specify whether to use Transfer-model InceptionResnetV2 or not')


def main(argv):
    # setup wandb
    project = "diabetic_retinopathy"
    wandb.init(project=project)

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # Check if multiple models were chosen
    if sum([FLAGS.vgg, FLAGS.resnet18, FLAGS.resnet34, FLAGS.mobilenet_v2, FLAGS.inception_v3,
            FLAGS.inception_resnet_v2]) > 1:
        print("ERROR: Please specify at most one model.")
        exit()

    # model
    if FLAGS.vgg:
        model = vgg_like(input_shape=ds_info["image_shape"], n_classes=ds_info["n_classes"])
    elif FLAGS.resnet18:
        model = resnet18()
        model.build(input_shape=(None,) + ds_info["image_shape"])
        model.summary()
    elif FLAGS.resnet34:
        model = resnet34()
        model.build(input_shape=(None,) + ds_info["image_shape"])
        model.summary()
    elif FLAGS.mobilenet_v2:
        model = MyMobileNetV2(n_classes=ds_info["n_classes"])
    elif FLAGS.inception_v3:
        model = inception_v3()
    elif FLAGS.inception_resnet_v2:
        model = inception_resnet_v2()
    else:
        print("ERROR: Please specify a model.")
        exit()

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
                 )

    if FLAGS.visualize:
        visualize(model=model, checkpoint=trainer.ckpt, manager=trainer.manager, img_height=ds_info["image_shape"][0],
                  img_width=ds_info["image_shape"][1], sobel_preprocessing=ds_info["sobel_preprocessing"])

    if FLAGS.visualize_an:
        visualize_an(model=model, checkpoint=trainer.ckpt, manager=trainer.manager, img_height=ds_info["image_shape"][0],
                  img_width=ds_info["image_shape"][1], sobel_preprocessing=ds_info["sobel_preprocessing"])
    wandb.finish()


if __name__ == "__main__":
    app.run(main)