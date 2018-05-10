package(default_visibility = ["//third_party/py/basenji:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "job_params_files",
    srcs = [
        "params.dilated_res.262k.txt",
        "params.small.131k.tfrecord.txt",
        "params.small.hd5.txt",
        "params.small.tfrecord.txt",
    ],
)

filegroup(
    name = "targets_tfrecords",
    srcs = glob(["targets32/**"]) + glob(["targets1/**"]),
)

filegroup(
    name = "cage_tfr",
    srcs = glob(["cage/tfrecord/**"]),
)

filegroup(
    name = "cage_h5",
    srcs = glob(["cage/h5/**"]),
)
