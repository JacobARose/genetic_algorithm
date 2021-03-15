from pyleaves.utils import set_tf_config
set_tf_config(num_gpus=1)

from pprint import pprint as pp
import os
import tensorflow as tf
from genetic_algorithm.models.zoo.resnet.resnet_v2_c import ResNetV2

from contrastive_learning.data.extant import NUM_CLASSES as num_classes, NUM_SAMPLES as num_samples

hyperparameters = {'weights':'imagenet'}

dataset_name='Extant'
model_name = 'resnet_50_v2'
target_size = (256,256,3)
val_split = 0.2
batch_size = 32
seed = 457

initial_frozen_layers = (0,-17)

resnet_tf_imagenet = ResNetV2.from_tf_pretrained(model_name=model_name,
                                                 input_shape=target_size, 
                                                 include_top=False,
                                                 num_classes=num_classes,
                                                 **hyperparameters)


save_dir = f'/media/data/jacob/GitHub/genetic_algorithm/tests/example_results/{dataset_name}_family_10/{resnet_tf_imagenet.model_name}_tf_imagenet/{dataset_name}__res{target_size[0]}/init_frozen_layers={initial_frozen_layers}'
os.makedirs(save_dir, exist_ok=True)

print(f'save_dir = {save_dir}')
pp(['num_classes:', num_classes, 'num_samples', num_samples])


resnet_tf_imagenet.fit_paleoai_dataset(dataset_name='Extant',
                                       target_size=target_size[:2],
                                       batch_size=batch_size,
                                       num_epochs=20,
                                       decay=('cosine', 0),
                                       initial_frozen_layers=initial_frozen_layers,
                                       save=save_dir,
                                       allow_resume=True, 
                                       seed=seed)