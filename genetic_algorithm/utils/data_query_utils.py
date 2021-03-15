# Created 12/10/2020 by Jacob A Rose



# from pyleaves.utils.WandB_artifact_utils import load_dataset_from_artifact, load_Leaves_Minus_PNAS_dataset, load_Leaves_Minus_PNAS_test_dataset



    
    
    
    


# def load_dataset_from_artifact(dataset_name='Fossil', threshold=4, test_size=0.3, version='latest', artifact_name=None, run=None):
#     train_size = 1 - test_size
#     if artifact_name:
#         pass
#     elif dataset_name=='Fossil':
#         artifact_name = f'{dataset_name}_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name=='PNAS':
#         artifact_name = f'{dataset_name}_family_{threshold}_50-50:{version}'
#     elif dataset_name=='Leaves':
#         artifact_name = f'{dataset_name}_family_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name=='Leaves-PNAS':
#         artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name in ['Leaves_in_PNAS', 'PNAS_in_Leaves']:
#         artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
            
#     artifact_uri = f'brown-serre-lab/paleoai-project/{artifact_name}'
#     return load_train_test_artifact(artifact_uri=artifact_uri, run=run)
    
    
    
    
#     if dataset_name == "Leaves-PNAS":
#         print('Loading Leaves-PNAS dataset for train/val, and loading PNAS_test for test')
#         train_df, val_df = load_dataset_from_artifact(dataset_name=dataset_name, threshold=threshold, test_size=test_size, version=version, artifact_name=artifact_name)
#         _, test_df = load_dataset_from_artifact(dataset_name='PNAS', threshold=100, test_size=0.5, version='latest')


#     else:
#         print(f'Loading {dataset_name} dataset for train and test, with train set split into {1-validation_split}:{validation_split} train:val subsets.')
#         train_df, test_df = load_dataset_from_artifact(dataset_name=dataset_name, threshold=threshold, test_size=test_size, version=version, artifact_name=artifact_name)
#         train_df, val_df = train_test_split(train_df, test_size=validation_split, random_state=seed, shuffle=True, stratify=train_df.family)
        
        
        
#     train_data_info = data_df_2_tf_data(train_df,
#                                         x_col='archive_path',
#                                         y_col='family',
#                                         training=True,
#                                         preprocess_input=preprocess_input,
#                                         seed=seed,
#                                         target_size=target_size,
#                                         batch_size=batch_size,
#                                         augmentations=augmentations,
#                                         num_parallel_calls=num_parallel_calls,
#                                         cache=False,
#                                         shuffle_first=True,
#                                         fit_class_weights=fit_class_weights,
#                                         subset_key='train',
#                                         use_tfrecords=use_tfrecords,
#                                         samples_per_shard=samples_per_shard,
#                                         tfrecord_dir=tfrecord_dir)
    

    
    
    
    
    
    
    
# def load_data_from_tensor_slices(data: pd.DataFrame, cache_paths: Union[bool,str]=True, training=False, seed=None, x_col='path', y_col='label', dtype=None):
#     import tensorflow as tf
#     dtype = dtype or tf.uint8
#     num_samples = data.shape[0]

#     def load_img(image_path):
#         img = tf.io.read_file(image_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.convert_image_dtype(img, tf.float32)
#         return img

#     x_data = tf.data.Dataset.from_tensor_slices(data[x_col].values.tolist())
#     y_data = tf.data.Dataset.from_tensor_slices(data[y_col].values.tolist())
#     data = tf.data.Dataset.zip((x_data, y_data))

#     data = data.take(num_samples)
    
#     # TODO TEST performance and randomness of the order of shuffle and cache when shuffling full dataset each iteration, but only filepaths and not full images.
#     if training:
#         data = data.shuffle(num_samples,seed=seed, reshuffle_each_iteration=True)
#     if cache_paths:
#         if isinstance(cache_paths, str):
#             data = data.cache(cache_paths)
#         else:
#             data = data.cache()

#     data = data.map(lambda x,y: (tf.image.convert_image_dtype(load_img(x)*255.0,dtype=dtype),y), num_parallel_calls=-1)
#     return data

