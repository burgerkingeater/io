# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AvroDataset"""

import sys
import uuid

import tensorflow as tf
from tensorflow_io.core.python.ops import core_ops


class _AvroIODatasetFunction:
    def __init__(self, function, resource, component, shape, dtype):
        self._function = function
        self._resource = resource
        self._component = component
        self._shape = tf.TensorShape([None]).concatenate(shape[1:])
        self._dtype = dtype

    def __call__(self, start, stop):
        return self._function(
            self._resource,
            start=start,
            stop=stop,
            component=self._component,
            shape=self._shape,
            dtype=self._dtype,
        )


class AvroIODataset(tf.compat.v2.data.Dataset):
    """AvroIODataset"""

    def __init__(self, filename, schema, columns=None, internal=True):
        """AvroIODataset."""
        if not internal:
            raise ValueError(
                "AvroIODataset constructor is private; please use one "
                "of the factory methods instead (e.g., "
                "IODataset.from_avro())"
            )
        with tf.name_scope("AvroIODataset") as scope:
            capacity = 4096

            metadata = ["schema: %s" % schema]
            resource, columns_v = core_ops.io_avro_readable_init(
                filename,
                metadata=metadata,
                container=scope,
                shared_name="{}/{}".format(filename, uuid.uuid4().hex),
            )
            columns = columns if columns is not None else columns_v.numpy()

            columns_dataset = []

            columns_function = []
            for column in columns:
                shape, dtype = core_ops.io_avro_readable_spec(resource, column)
                shape = tf.TensorShape([None if e < 0 else e for e in shape.numpy()])
                dtype = tf.as_dtype(dtype.numpy())
                function = _AvroIODatasetFunction(
                    core_ops.io_avro_readable_read, resource, column, shape, dtype
                )
                columns_function.append(function)

            for (column, function) in zip(columns, columns_function):
                column_dataset = tf.compat.v2.data.Dataset.range(
                    0, sys.maxsize, capacity
                )
                column_dataset = column_dataset.map(
                    lambda index: function(index, index + capacity)
                )
                column_dataset = column_dataset.apply(
                    tf.data.experimental.take_while(
                        lambda v: tf.greater(tf.shape(v)[0], 0)
                    )
                )
                columns_dataset.append(column_dataset)
            if len(columns_dataset) == 1:
                dataset = columns_dataset[0]
            else:
                dataset = tf.compat.v2.data.Dataset.zip(tuple(columns_dataset))
            dataset = dataset.unbatch()

            self._function = columns_function
            self._dataset = dataset
            super().__init__(
                self._dataset._variant_tensor
            )  # pylint: disable=protected-access

    def _inputs(self):
        return []

    @property
    def element_spec(self):
        return self._dataset.element_spec


def make_avro_dataset(
    filenames,
    reader_schema,
    features,
    batch_size,
    num_epochs,
    num_parallel_calls=2,
    label_keys=None,
    input_stream_buffer_size=16 * 1024,
    avro_data_buffer_size=256,
    shuffle=True,
    shuffle_buffer_size=10000,
    shuffle_seed=None,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    num_parallel_reads=1,
):
    """Makes an avro dataset.
    Reads from avro files and parses the contents into tensors.
    Args:
      filenames: A `tf.string` tensor containing one or more filenames.
      reader_schema: A `tf.string` scalar for schema resolution.
      features: Is a map of keys that describe a single entry or sparse vector
                in the avro record and map that entry to a tensor. The syntax
                is as follows:
                features = {'my_meta_data.size':
                            tf.FixedLenFeature([], tf.int64)}
                Select the 'size' field from a record metadata that is in the
                field 'my_meta_data'. In this example we assume that the size is
                encoded as a long in the Avro record for the metadata.
                features = {'my_map_data['source'].ip_addresses':
                            tf.VarLenFeature([], tf.string)}
                Select the 'ip_addresses' for the 'source' key in the map
                'my_map_data'. Notice we assume that IP addresses are encoded as
                strings in this example.
                features = {'my_friends[1].first_name':
                            tf.FixedLenFeature([], tf.string)}
                Select the 'first_name' for the second friend with index '1'.
                This assumes that all of your data has a second friend. In
                addition, we assume that all friends have only one first name.
                For this reason we chose a 'FixedLenFeature'.
                features = {'my_friends[*].first_name':
                            tf.VarLenFeature([], tf.string)}
                Selects all first_names in each row. For this example we use the
                wildcard '*' to indicate that we want to select all 'first_name'
                entries from the array.
                features = {'sparse_features':
                            tf.SparseFeature(index_key='index',
                                             value_key='value',
                                             dtype=tf.float32, size=10)}
                We assume that sparse features contains an array with records
                that contain an 'index' field that MUST BE LONG and an 'value'
                field with floats (single precision).
      batch_size: Items in a batch, must be > 0
      num_parallel_calls: Number of parallel calls
      label_key: The label key, if None no label will be returned
      num_epochs: The number of epochs. If number of epochs is set to None we
                  cycle infinite times and drop the remainder automatically.
                  This will make all batch sizes the same size and static.
      input_stream_buffer_size: The size of the input stream buffer in By
      avro_data_buffer_size: The size of the avro data buffer in By
    """
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        n_filenames = tf.shape(filenames, out_type=tf.int64)[0]
        dataset = dataset.shuffle(n_filenames, shuffle_seed)
    if label_keys is None:
        label_keys = []
    # Handle the case where the user only provided a single label key
    if not isinstance(label_keys, list):
        label_keys = [label_keys]
    for label_key in label_keys:
        if label_key not in features:
            raise ValueError(
                "`label_key` provided (%r) must be in `features`." % label_key
            )
    def filename_to_dataset(filename):
        # Batches
        return _AvroDataset(
            filenames=filename,
            features=features,
            reader_schema=reader_schema,
            batch_size=batch_size,
            drop_remainder=num_epochs is None,
            num_parallel_calls=num_parallel_calls,
            input_stream_buffer_size=input_stream_buffer_size,
            avro_data_buffer_size=avro_data_buffer_size,
        )
    # Read files sequentially (if num_parallel_reads=1) or in parallel
    dataset = dataset.interleave(
        filename_to_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_reads,
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, shuffle_seed)
    if num_epochs != 1:
        dataset = dataset.repeat(num_epochs)
    if any(isinstance(feature, tf.io.SparseFeature) for _, feature in features.items()):
        # pylint: disable=protected-access
        # pylint: disable=g-long-lambda
        dataset = dataset.map(
            lambda x: construct_tensors_for_composite_features(features, x),
            num_parallel_calls=num_parallel_calls,
        )
    # Take care of sparse shape assignment in features
    def reshape_sp_function(tensor_features):
        """Note, that sparse merge produces a rank of 2*n instead of n+1 when
        merging n dimensional tensors. But the index is produced with rank n+1.
        We correct the shape here through this method.
        :param tensor_features: the output features dict from avrodataset
        """
        for feature_name, feature in features.items():
            if (
                isinstance(feature, tf.io.SparseFeature)
                and isinstance(feature.size, list)
                and len(feature.size) > 1
            ):
                # Have -1 for unknown batch
                reshape = [-1] + list(feature.size)
                tensor_features[feature_name] = tf.sparse.reshape(
                    tensor_features[feature_name], reshape
                )
        return tensor_features
    dataset = dataset.map(reshape_sp_function, num_parallel_calls=num_parallel_calls)
    if len(label_keys) > 0:
        dataset = dataset.map(
            lambda x: (x, {label_key_: x.pop(label_key_) for label_key_ in label_keys}),
            num_parallel_calls=num_parallel_calls,
        )
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset