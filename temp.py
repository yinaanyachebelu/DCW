import torch
import tensorflow as tf

torch.cuda.is_available()


print(tf.test.gpu_device_name())
print("device is ", torch.device)
