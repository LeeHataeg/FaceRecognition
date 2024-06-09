import os

from bing_image_downloader.downloader import download

datasets_path = os.path.join(os.getcwd(), "dataset")

for i in ["배우 공유"]:
    download(
        i,
        limit=10,
        output_dir="star",
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True,
    )
