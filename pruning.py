#!/usr/bin/env python
import tensorflow as tf
import datetime
import numpy as np
import pathlib
import time
import logging
import os
import tf2onnx
import tensorflow_model_optimization as tfmot

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s - %(message)s')

# Findings: post training quantization does not support dynamic batch sizes, that means 
# fix the input shape but it needs to adhere to that shape, also some constraints on speedup claims, int8 does not provide GPU acceleration
# Also haven't figured out a way to convert the model to onnx to maybe use tensorrt and see its gains
# ONNX runtime is giving close to 7x speedup just based on the raw example out of the box on CPU
# benchmarks done for BENCHMARK_RUNS for model() and sess.run inside onnxruntime

class EpochTrainingTime(tf.keras.callbacks.Callback):
	def __init__(self):
		self.epoch_times = []
	def on_epoch_begin(self, epoch, logs=None):
		self.start_time = time.time()
	def on_epoch_end(self, epoch, logs=None):
		self.epoch_times.append(time.time() - self.start_time)

pathlib.Path('/tmp/checkpoint/').mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
IMAGE_SHAPE = (32, 32, 3)
CLASSES = 10
EPOCHS = 1
CHECKPOINT_PATH = '/tmp/checkpoint/ckpt-{epoch:02d}-{val_loss:.2f}'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
lr = 0.0001
MODEL_PATH = 'model.onnx'
BENCHMARK_RUNS = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def get_dataset():
	train_dataset = \
		tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
	return train_dataset, test_dataset

def train_base_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(32, 3, padding='same', input_shape=(32,32,3), activation='relu'),
		tf.keras.layers.Conv2D(32, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.25),

		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(64, 3, activation='relu'),
		tf.keras.layers.MaxPooling2D(),
		tf.keras.layers.Dropout(0.25),

		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(CLASSES, activation='softmax'),
	])

	latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)

	model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
				metrics=['accuracy'])

	if latest:
		logging.info(f'Loading from checkpoint file: {latest}')
		model.load_weights(latest)

	model.summary()

	log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
															save_weights_only=True, save_best_weights=True, monitor="val_accuracy")

	train_dataset, test_dataset = get_dataset()

	# train the model
	history = model.fit(train_dataset,
						epochs=EPOCHS,
						validation_data=test_dataset,
						callbacks=[tensorboard_callback, checkpoint_callback])
	return model

def convert_keras_to_onnx(model: tf.keras.Model, output_path: str):
	_, _ = tf2onnx.convert.from_keras(model, output_path=output_path)

def benchmark_model(model: tf.keras.Model, dataset: np.ndarray) -> None:
	execution_times = []
	for i in range(BENCHMARK_RUNS):
		begin_time = time.time()
		model.predict_on_batch(dataset[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
		end_time = time.time()
		execution_times.append(end_time - start_time)
	logging.info(f'Execution times for {BENCHMARK_RUNS} runs: {execution_times}')
	logging.info(f'(KERAS) Elapsed average time: {sum(execution_times)/BENCHMARK_RUNS}')


def benchmark_onnx_vs_keras_inference(model_path):
	import onnx
	import onnxruntime as ort
	import numpy as np
	onnx_model = onnx.load(model_path)
	onnx.checker.check_model(onnx_model)
	ort_session = ort.InferenceSession(model_path)
	x = np.expand_dims(x_train[0], axis=0).astype(np.float32)
	total_time = 0
	for _ in range(BENCHMARK_RUNS):
		start_time = time.time()
		outputs = ort_session.run(None, {'conv2d_input': x_test[:32].astype(np.float32)})
		total_time += (time.time() - start_time)
	logging.info(f'(ONNXRUNTIME) Elapsed time: {total_time / BENCHMARK_RUNS}')

	benchmark_model(model, x_test)

# benchmark_onnx_vs_keras_inference(MODEL_PATH)

# lets do some pruning
# base sparsity pruning did not give any speedups compared to the dense model, that's a bummer
def sparsify(model: tf.keras.Model) -> None:
	pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)
	pruned_model.summary()
	callbacks = [
		tfmot.sparsity.keras.UpdatePruningStep(),
		tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
	]
	pruned_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
						loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
						metrics=['accuracy'])


	pruned_model.fit(train_dataset,
					epochs=2,
					validation_data=test_dataset,
					callbacks=callbacks)
	
	benchmark_model(pruned_model, x_test)

import torchvision
import torch.nn as nn
vgg = torchvision.models.vgg16(pretrained=True)
for name, module in vgg.named_modules():
	print(type(module))