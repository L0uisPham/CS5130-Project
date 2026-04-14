import tensorflow_datasets as tfds

manual_dir = tfds.core.utils.gcs_utils.expand_path(
    "~/tensorflow_datasets/downloads/manual"
)
builder = tfds.builder(
    "chexpert",
    download_config=tfds.download.DownloadConfig(manual_dir=manual_dir),
)
builder.download_and_prepare()
print("Prepared at:", builder.data_dir)
print(builder.info)
