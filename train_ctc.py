import os
import re
import argparse
import pprint
import shutil
import tensorflow as tf
import yaml
from pathlib import Path
import cv2

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError:
    # tf < 2.4.0
    AUTOTUNE = tf.data.experimental.AUTOTUNE


class Dataset(tf.data.TextLineDataset):
    def __init__(self, filename, **kwargs):
        self.dirname = os.path.dirname(filename)
        super().__init__(filename, **kwargs)

    def parse_func(self, line):
        raise NotImplementedError

    def parse_line(self, line):
        line = tf.strings.strip(line)
        img_relative_path, label = self.parse_func(line)
        img_path = tf.strings.join([self.dirname, os.sep, img_relative_path])
        return img_path, label


class SimpleDataset(Dataset):
    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path, label = splited_line[0], splited_line[1]
        return img_relative_path, label


class MJSynthDataset(Dataset):
    def parse_func(self, line):
        splited_line = tf.strings.split(line)
        img_relative_path = splited_line[0]
        label = tf.strings.split(img_relative_path, sep="_")[1]
        return img_relative_path, label


class ICDARDataset(Dataset):
    def parse_func(self, line):
        splited_line = tf.strings.split(line, sep=",")
        img_relative_path, label = splited_line[0], splited_line[1]
        label = tf.strings.strip(label)
        label = tf.strings.regex_replace(label, r'"', "")
        return img_relative_path, label


class DatasetBuilder:
    def __init__(
        self,
        table_path,
        img_shape=(32, None, 3),
        max_img_width=300,
        ignore_case=False,
    ):
        # map unknown label to 0
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                table_path,
                tf.string,
                tf.lookup.TextFileIndex.WHOLE_LINE,
                tf.int64,
                tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            0,
        )
        self.img_shape = img_shape
        self.ignore_case = ignore_case
        if img_shape[1] is None:
            self.max_img_width = max_img_width
            self.preserve_aspect_ratio = True
        else:
            self.preserve_aspect_ratio = False

    @property
    def num_classes(self):
        return self.table.size()

    def _parse_annotation(self, path):
        with open(path) as f:
            line = f.readline().strip()
        if re.fullmatch(r".*/*\d+_.+_(\d+)\.\w+ \1", line):
            return MJSynthDataset(path)
        elif re.fullmatch(r'.*/*word_\d\.\w+, ".+"', line):
            return ICDARDataset(path)
        elif re.fullmatch(r".+\.\w+ .+", line):
            return SimpleDataset(path)
        else:
            raise ValueError("Unsupported annotation format")

    def _concatenate_ds(self, ann_paths):
        datasets = [self._parse_annotation(path) for path in ann_paths]
        concatenated_ds = datasets[0].map(datasets[0].parse_line)
        for ds in datasets[1:]:
            ds = ds.map(ds.parse_line)
            concatenated_ds = concatenated_ds.concatenate(ds)
        return concatenated_ds

    def _decode_img(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_shape[-1])
        if self.preserve_aspect_ratio:
            img_shape = tf.shape(img)
            scale_factor = self.img_shape[0] / img_shape[0]
            img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
            img_width = tf.cast(img_width, tf.int32)
        else:
            img_width = self.img_shape[1]
        img = tf.image.resize(img, (self.img_shape[0], img_width)) / 255.0
        return img, label

    def _filter_img(self, img, label):
        img_shape = tf.shape(img)
        return img_shape[1] < self.max_img_width

    def _tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, "UTF-8")
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        # TODO(hym) Waiting for official support to use RaggedTensor in keras
        tokens = tokens.to_sparse()
        return imgs, tokens

    def __call__(self, ann_paths, batch_size, is_training):
        ds = self._concatenate_ds(ann_paths)
        if self.ignore_case:
            ds = ds.map(lambda x, y: (x, tf.strings.lower(y)))
        if is_training:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self._decode_img, AUTOTUNE)
        if self.preserve_aspect_ratio and batch_size != 1:
            ds = ds.filter(self._filter_img)
            ds = ds.padded_batch(batch_size, drop_remainder=is_training)
        else:
            ds = ds.batch(batch_size, drop_remainder=is_training)
        ds = ds.map(self._tokenize, AUTOTUNE)
        ds = ds.prefetch(AUTOTUNE)
        return ds


def vgg_style(x):
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv1")(
        x
    )
    x = tf.keras.layers.MaxPool2D(pool_size=2, padding="same", name="pool1")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2")(
        x
    )
    x = tf.keras.layers.MaxPool2D(pool_size=2, padding="same", name="pool2")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(
        x
    )
    x = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=(2, 1), padding="same", name="pool4"
    )(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", use_bias=False, name="conv5")(x)
    x = tf.keras.layers.BatchNormalization(name="bn5")(x)
    x = tf.keras.layers.Activation("relu", name="relu5")(x)
    x = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv6")(
        x
    )
    x = tf.keras.layers.MaxPool2D(
        pool_size=2, strides=(2, 1), padding="same", name="pool6"
    )(x)
    x = tf.keras.layers.Conv2D(512, 2, use_bias=False, name="conv7")(x)
    x = tf.keras.layers.BatchNormalization(name="bn7")(x)
    x = tf.keras.layers.Activation("relu", name="relu7")(x)
    x = tf.keras.layers.Reshape((-1, 512), name="reshape7")(x)
    return x


def build_model(
    num_classes,
    weight=None,
    preprocess=None,
    postprocess=None,
    img_shape=(64, 512, 3),  # Update the input shape
    model_name="crnn",
):
    x = img_input = tf.keras.Input(shape=img_shape)
    if preprocess is not None:
        x = preprocess(x)

    x = vgg_style(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True), name="bi_lstm1"
    )(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(units=256, return_sequences=True), name="bi_lstm2"
    )(x)
    x = tf.keras.layers.Dense(units=num_classes, name="logits")(x)

    if postprocess is not None:
        x = postprocess(x)

    model = tf.keras.Model(inputs=img_input, outputs=x, name=model_name)
    if weight is not None:
        model.load_weights(weight, by_name=True, skip_mismatch=True)
    return model


class SequenceAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="sequence_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        def sparse2dense(tensor, shape):
            tensor = tf.sparse.reset_shape(tensor, shape)
            tensor = tf.sparse.to_dense(tensor, default_value=-1)
            tensor = tf.cast(tensor, tf.float32)
            return tensor

        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        y_pred_shape = tf.shape(y_pred)
        max_width = tf.math.maximum(y_true_shape[1], y_pred_shape[1])
        logit_length = tf.fill([batch_size], y_pred_shape[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length,
        )
        y_true = sparse2dense(y_true, [batch_size, max_width])
        y_pred = sparse2dense(decoded[0], [batch_size, max_width])
        num_errors = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
        num_errors = tf.cast(num_errors, tf.float32)
        num_errors = tf.math.reduce_sum(num_errors)
        batch_size = tf.cast(batch_size, tf.float32)
        self.total.assign_add(batch_size)
        self.count.assign_add(batch_size - num_errors)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)


class EditDistance(tf.keras.metrics.Metric):
    def __init__(self, name="edit_distance", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.sum_distance = self.add_weight(name="sum_distance", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_shape = tf.shape(y_pred)
        batch_size = y_pred_shape[0]
        logit_length = tf.fill([batch_size], y_pred_shape[1])
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length,
        )
        sum_distance = tf.math.reduce_sum(tf.edit_distance(decoded[0], y_true))
        batch_size = tf.cast(batch_size, tf.float32)
        self.sum_distance.assign_add(sum_distance)
        self.total.assign_add(batch_size)

    def result(self):
        return self.sum_distance / self.total

    def reset_states(self):
        self.sum_distance.assign(0)
        self.total.assign(0)


class CTCLoss(tf.keras.losses.Loss):
    """A class that wraps the function of tf.nn.ctc_loss.

    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits],
            If True, logits is shaped [time, batch, logits].
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1).
    """

    def __init__(self, logits_time_major=False, blank_index=-1, name="ctc_loss"):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        """Computes CTC (Connectionist Temporal Classification) loss. work on
        CPU, because y_true is a SparseTensor.
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred_shape = tf.shape(y_pred)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index,
        )
        return tf.math.reduce_mean(loss)


class CTCDecoder(tf.keras.layers.Layer):
    def __init__(self, table_path, **kwargs):
        super().__init__(**kwargs)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.TextFileInitializer(
                table_path,
                tf.int64,
                tf.lookup.TextFileIndex.LINE_NUMBER,
                tf.string,
                tf.lookup.TextFileIndex.WHOLE_LINE,
            ),
            "",
        )

    def detokenize(self, x):
        x = tf.RaggedTensor.from_sparse(x)
        x = tf.ragged.map_flat_values(self.table.lookup, x)
        strings = tf.strings.reduce_join(x, axis=1)
        return strings


class CTCGreedyDecoder(CTCDecoder):
    def __init__(self, table_path, merge_repeated=True, **kwargs):
        super().__init__(table_path, **kwargs)
        self.merge_repeated = merge_repeated

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        sequence_length = tf.fill([input_shape[0]], input_shape[1])
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]),
            sequence_length,
            self.merge_repeated,
        )
        strings = self.detokenize(decoded[0])
        labels = tf.cast(decoded[0], tf.int32)
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=inputs,
            label_length=None,
            logit_length=sequence_length,
            logits_time_major=False,
            blank_index=-1,
        )
        probability = tf.math.exp(-loss)
        return strings, probability


class CTCBeamSearchDecoder(CTCDecoder):
    def __init__(self, table_path, beam_width=100, top_paths=1, **kwargs):
        super().__init__(table_path, **kwargs)
        self.beam_width = beam_width
        self.top_paths = top_paths

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        decoded, log_probability = tf.nn.ctc_beam_search_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]),
            tf.fill([input_shape[0]], input_shape[1]),
            self.beam_width,
            self.top_paths,
        )
        strings = []
        for i in range(self.top_paths):
            strings.append(self.detokenize(decoded[i]))
        strings = tf.concat(strings, 1)
        probability = tf.math.exp(log_probability)
        return strings, probability


# Configuration
config = {
    "train": {
        "dataset_builder": {
            "table_path": "labels.txt",
            "img_shape": [64, 512, 3],  # Updated input size
            "max_img_width": 512,
            "ignore_case": True,
        },
        "train_ann_paths": ["train_annotation.txt"],
        "val_ann_paths": ["val_annotation.txt"],
        "batch_size_per_replica": 256,
        "epochs": 20,
        "lr_schedule": {
            "initial_learning_rate": 0.0001,
            "decay_steps": 600000,
            "alpha": 0.01,
        },
        "tensorboard": {
            "histogram_freq": 1,
            "profile_batch": 0,
        },
    },
}

# Load configuration
train_config = config["train"]
dataset_builder = DatasetBuilder(**train_config["dataset_builder"])

# Create datasets
batch_size = train_config["batch_size_per_replica"]
train_ds = dataset_builder(train_config["train_ann_paths"], batch_size, True)
val_ds = dataset_builder(train_config["val_ann_paths"], batch_size, False)

# Distributed training strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Build and compile the model
    model = build_model(
        dataset_builder.num_classes,
        img_shape=train_config["dataset_builder"]["img_shape"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            tf.keras.optimizers.schedules.CosineDecay(**train_config["lr_schedule"])
        ),
        loss=CTCLoss(),
        metrics=[SequenceAccuracy()],
    )

# Model summary
model.summary()

# Model training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "checkpoints/{epoch}.h5", save_weights_only=True
    ),
    tf.keras.callbacks.TensorBoard(log_dir="logs", **train_config["tensorboard"]),
]

model.fit(
    train_ds,
    epochs=train_config["epochs"],
    callbacks=callbacks,
    validation_data=val_ds,
)
