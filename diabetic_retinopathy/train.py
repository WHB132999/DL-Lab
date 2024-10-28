import datetime
import gin
import tensorflow as tf
import logging
import wandb

# CREATED BY HUIBIN WANG AND SAMUEL BRUCKER

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, total_steps, log_interval, ckpt_interval,
                 restore_from_checkpoint):

        self.restore = restore_from_checkpoint

        # Summary Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # Checkpoint Manager
        # ... https://www.tensorflow.org/guide/checkpoint
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                        model=model)
        self.manager = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=10)

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self):

        if self.restore is True:
            self.ckpt.restore(self.manager.latest_checkpoint)
            if self.manager.latest_checkpoint:
                print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            self.ckpt.step.assign_add(1)

            # write train summary to tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=step)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)  # self.epoch)

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for test_images, test_labels in self.ds_val:
                    self.test_step(test_images, test_labels)
                # write test summary to tensorboard
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)  #

                template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.test_loss.result(),
                                             self.test_accuracy.result() * 100))

                # wandb logging
                wandb.log({'train_loss': self.train_loss.result(),
                           'train_acc': self.train_accuracy.result() * 100,
                           'val_loss': self.test_loss.result(),
                           'val_acc': self.test_accuracy.result() * 100,
                           'step': step})

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                save_path = self.manager.save()
                print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(self.train_loss.result().numpy()))

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                save_path = self.manager.save()
                print("Saved checkpoint for FINAL step {}: {}".format(int(self.ckpt.step), save_path))
                print("loss {:1.2f}".format(self.train_loss.result().numpy()))

                return self.test_accuracy.result().numpy()