import math
import os
import requests
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm
from urllib.parse import urlparse

def sizeof_fmt(size, suffix='B'):
    """Get human readable file size.

    Args:
        size (int): File size.
        suffix (str): Suffix. Default: 'B'.

    Return:
        str: Formated file siz.
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f'{size:3.1f} {unit}{suffix}'
        size /= 1024.0
    return f'{size:3.1f} Y{suffix}'


def download_file_from_google_drive(file_id, save_path):
    """Download files from google drive.
    Ref:
    https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive  # noqa E501
    Args:
        file_id (str): File id.
        save_path (str): Save path.
    """

    session = requests.Session()
    URL = 'https://docs.google.com/uc?export=download'
    params = {'id': file_id}

    response = session.get(URL, params=params, stream=True)
    token = get_confirm_token(response)
    if token:
        params['confirm'] = token
        response = session.get(URL, params=params, stream=True)

    # get file size
    response_file_size = session.get(URL, params=params, stream=True, headers={'Range': 'bytes=0-2'})
    print(response_file_size)
    if 'Content-Range' in response_file_size.headers:
        file_size = int(response_file_size.headers['Content-Range'].split('/')[1])
    else:
        file_size = None

    save_response_content(response, save_path, file_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination, file_size=None, chunk_size=32768):
    if file_size is not None:
        pbar = tqdm(total=math.ceil(file_size / chunk_size), unit='chunk')

        readable_file_size = sizeof_fmt(file_size)
    else:
        pbar = None

    with open(destination, 'wb') as f:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size):
            downloaded_size += chunk_size
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f'Download {sizeof_fmt(downloaded_size)} / {readable_file_size}')
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
        if pbar is not None:
            pbar.close()


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """在 `model_dir` 里找 `url` 对应的权重文件并返回**本地绝对路径**。

    ⚠️ 本工程把它改成了**只查本地、绝不下载**。

    上游的实现是「本地没有就从 GitHub Releases 拉」—— 那会在缺权重时静默联网。
    Detext 是自包含的：所有模型都在 `models/` 下，不希望任何一次推理触发外网请求
    （离线机器上会卡到超时，有网机器上会悄悄引入版本不一致的权重）。
    所以这里删掉了 `download_url_to_file` 分支，改成缺文件就大声报错。

    调用方（`propainter/inference.py`）仍然传 url，我们只用它取文件名 ——
    `weights/propainter/` 下那三个 .pth 就是按这个名字放好的。
    """
    if model_dir is None:  # 上游的 fallback，本工程不用
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        raise FileNotFoundError(
            f"缺少先验模型权重 {filename}\n"
            f"  期望位置: {cached_file}\n"
            f"  本工程不会联网下载（原始 URL 是 {url}）。\n"
            f"  请把该文件放进 models/propainter/ 后重试。")
    return cached_file