from tensorpack import *
from tensorpack.tfutils.sessinit import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow

from preprocess import get_train_test_data
from vgg import VGGModel

def batch_data(data, augment=False):
    if augment:
      augmentors = [
        imgaug.Rotation(max_deg = 20),
        # imgaug.GaussianBlur(max_size=2),
        imgaug.Shift()
      ]
      data = AugmentImageComponent(data, augmentors)
    batch_size = 50
    data = BatchData(data, 50, remainder=False)
    return data

if __name__ == '__main__':
    train_data, test_data = get_train_test_data()
    batched_train = batch_data(train_data)
    batched_test = batch_data(test_data)
    config = TrainConfig(
        VGGModel(),
        dataflow=batched_train,
        callbacks=[
            # save the current model
            ModelSaver(),
            # evaluate the current model and print out the loss
            InferenceRunner(batched_test,
                            [ScalarStats('cost'), ClassificationError()])
            # callbacks here to change hyperparameters
        ],
        max_epoch=hp.num_epochs,
        nr_tower=max(get_nr_gpu(), 1),
        session_init=None
    )
    # TensorPack: Training with simple one at a time feed into batches
    launch_train_with_config(config, SimpleTrainer())
