from dataclasses import dataclass

import tensorflow as tf
from keras_vggface.vggface import VGGFace
import torch
import torch.nn.functional as F

from framework.feature_extractor.feature_extractor_base import FeatureExtractorBase

@dataclass
class VGGFaceFE(FeatureExtractorBase):
    """
    VGG Face Feature Extractor
    """
    model : tf.keras.Model = None
    cuda : bool = False

    def __post_init__(self):
        self.name = 'VGG Face ResNet50'
        # set dynamic memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            self.cuda = True
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        with tf.device('/CPU:0'):
            # load VGG Face model
            vggface_model = VGGFace(model='resnet50', include_top=False)
            model_out = vggface_model.get_layer('avg_pool').output## create keras model
            self.model = tf.keras.Model(vggface_model.input, model_out)
            
    def extract(self, samples: torch.Tensor.type) -> torch.Tensor.type:
        # Upsample if necessary
        if samples.size(2) != 224 or samples.size(2) != 224:
            samples = F.interpolate(samples.float(), size=(224,224), mode="bicubic", align_corners=True)
        if self.cuda:
            with tf.device('/GPU:0'):
                # Pytorch -> Tensorflow
                tf_samples = tf.convert_to_tensor(samples.permute(0,2,3,1).cpu().numpy())
                # ResNet50 preprocessing
                tf_samples = tf.keras.applications.resnet50.preprocess_input(tf_samples)
                tf_features = self.model(tf_samples)
        else:
            tf_samples = tf.convert_to_tensor(samples.permute(0,2,3,1).numpy())
            # ResNet50 preprocessing
            tf_samples = tf.keras.applications.resnet50.preprocess_input(tf_samples)
            tf_features = self.model(tf_samples)
        # Tensorflow -> Pytorch
        torch_features = torch.from_numpy(tf_features.numpy()).squeeze()
        return torch_features