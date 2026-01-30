import tensorflow_datasets as tfds

manual_dir = tfds.core.utils.gcs_utils.expand_path(
    "~/tensorflow_datasets/downloads/manual"
)

ds_train = tfds.load(
    "chexpert",
    split="train",
    download=False,
    download_and_prepare_kwargs={
        "download_config": tfds.download.DownloadConfig(manual_dir=manual_dir)
    },
)

ds_valid = tfds.load(
    "chexpert",
    split="validation",
    download=False,
    download_and_prepare_kwargs={
        "download_config": tfds.download.DownloadConfig(manual_dir=manual_dir)
    },
)

print(ds_train)
print(ds_valid)
