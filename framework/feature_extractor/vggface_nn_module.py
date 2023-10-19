import tensorflow as tf
from keras_vggface import VGGFace
import torch
import torch.nn.functional as F

from framework.feature_extractor.vggface import VGGFaceFE



class VGGFaceNNModule(torch.nn.Module):
    """
    Torch NN Wrapper for VGG Face Feature Extractor
    Returning both Pool and Logits
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
            vggface_model = VGGFace(model='resnet50', include_top=True)
            pool_out = vggface_model.get_layer('avg_pool').output
            logits_out = vggface_model.get_layer('classifier').output
            ## create keras model
            self.model = tf.keras.Model(vggface_model.input, [pool_out, logits_out])

    def forward(self, samples):
        # Upsample if necessary
        if samples.size(2) != 224 or samples.size(3) != 224:
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
        torch_pool_features = torch.from_numpy(tf_features[0].numpy()).squeeze()
        torch_logits_features = torch.from_numpy(tf_features[1].numpy())
        return torch_pool_features, torch_logits_features

class VGGFaceNNModuleFeatures(torch.nn.Module):
    """
    Torch NN Wrapper for VGG Face Feature Extractor
    Returning only Pool activations
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = VGGFaceFE()

    def forward(self, samples):
        return self.model.extract(samples)