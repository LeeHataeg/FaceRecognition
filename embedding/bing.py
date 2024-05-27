from bing_image_downloader.downloader import download

for i in ["윤아"]:
    download(
        i,
        limit=300,
        output_dir="dataset",
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True,
    )
