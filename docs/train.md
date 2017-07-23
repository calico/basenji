### Basenji
###### Sequential regulatory activity predictions with deep convolutional neural networks.
--------------------------------------------------------------------------------
## Learn

<a name="train"/>

### basenji_train.py

Train a convolutional neural network to make sequential predictions on the given data.

| Argument | Type | Description |
| --- | --- | --- |
| params_file | Text table | Model configuration parameters. |
| data_file | HDF5 | Input training and validation data. |

The model should be trained on a GPU so that it runs at a reasonable pace. Whening assigning ops to devices, TensorFlow gives priority to your gpu:0 device (over cpu:0) if the GPU is available and supported.

To print whether the model is being trained on the GPU, run basenji_train.py with the `log_device_placement` flag set to `True`. In this sample output, training happens on the CPU (The GPU is unsupported.):

    Device mapping:
    ...
    2017-07-23 12:31:25.796354: I tensorflow/core/common_runtime/simple_placer.cc:847] cnn1/BatchNorm/Const: (Const)/job:localhost/replica:0/task:0/cpu:0
    cnn0/BatchNorm/Const_1: (Const): /job:localhost/replica:0/task:0/cpu:0
    2017-07-23 12:31:25.796361: I tensorflow/core/common_runtime/simple_placer.cc:847] cnn0/BatchNorm/Const_1: (Const)/job:localhost/replica:0/task:0/cpu:0
    cnn0/BatchNorm/Const: (Const): /job:localhost/replica:0/task:0/cpu:0
    2017-07-23 12:31:25.796368: I tensorflow/core/common_runtime/simple_placer.cc:847] cnn0/BatchNorm/Const: (Const)/job:localhost/replica:0/task:0/cpu:0
    Initialization time 15.614956