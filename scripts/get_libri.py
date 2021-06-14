import argparse
import fnmatch
import json
import logging
import os
import subprocess
import tarfile
import urllib.request
from functools import partial
import multiprocessing as mp
import sox
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LibriSpeech Data download')
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_sets", default="BASE", type=str, help="选择使用的数据集，BASE表示dev-clean和train-100, "
                                                                  "ALL表示所有数据，dev_clean等表示单个数据集")
parser.add_argument("--num_workers", default=6, type=int)
args = parser.parse_args()

URLS = {
    'TRAIN-CLEAN-100': "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    'TRAIN-CLEAN-360': "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    'TRAIN-OTHER-500': "http://www.openslr.org/resources/12/train-other-500.tar.gz",
    'DEV-CLEAN': "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    'DEV-OTHER': "http://www.openslr.org/resources/12/dev-other.tar.gz",
    'TEST-CLEAN': "http://www.openslr.org/resources/12/test-clean.tar.gz",
    'TEST-OTHER': "http://www.openslr.org/resources/12/test-other.tar.gz",
}
logging.basicConfig(level=logging.DEBUG)


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))
        urllib.request.urlretrieve(source, filename=destination + '.tmp')
        os.rename(destination + '.tmp', destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info('Not extracting. Maybe already there?')


def __process_data(data_folder: str, dst_folder: str, manifest_file: str, num_workers=1):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []

    for root, dirnames, filenames in os.walk(data_folder):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            files.append((os.path.join(root, filename), root))

    for transcripts_file, root in tqdm(files, total=len(files), desc="读取音频路径和文本"):
        with open(transcripts_file, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.lower().strip()
                flac_file = os.path.join(root, id + ".flac")
                entry = dict()
                entry['path'] = flac_file
                entry['text'] = transcript_text
                entries.append(entry)
    target_lists = transform_all_wavs(entries, target_wav_dir=dst_folder, num_workers=num_workers)
    with open(manifest_file, 'w', encoding='utf-8') as f:
        for json_line in target_lists:
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        logging.info("写入json文件完成: " + manifest_file)


def transform_all_wavs(wav_info: list, target_wav_dir, num_workers=6):
    """
    多进程的处理音频, 返回音频相关的信息
    Inputs:
        wav_info: 一个tsv文件中的音频信息
        target_wav_dir: 目标文件夹: .../dev
    """
    assert os.path.exists(target_wav_dir)
    # 组装音频为source.wav, target.wav 格式
    source_target_pack_wav_lists = []
    for item in wav_info:
        source_wav = item["path"]
        target_wav = os.path.abspath(os.path.join(target_wav_dir, source_wav.split('/')[-1].split('.')[0]+'.wav'))
        source_target_pack_wav_lists.append((source_wav, target_wav, item["text"]))
    processed_info = mp.Manager().list()  # 创建一个线程安全的列表,存放信息
    transform = sox.Transformer().convert(samplerate=16000, n_channels=1, bitdepth=16)
    partial_transform_wav = partial(transform_wav, transform=transform,
                                    processed_info=processed_info)
    import time
    pre_time = time.time()
    with mp.pool.Pool(num_workers) as pool:
        list(tqdm(pool.imap(partial_transform_wav, source_target_pack_wav_lists),
                  desc="转换音频", total=len(source_target_pack_wav_lists)))
    print("用时"+str(time.time()-pre_time))
    return list(processed_info)


def transform_wav(source_target_pack, transform, processed_info):
    source_wav, target_wav, text = source_target_pack
    if os.path.exists(target_wav):
        logging.warning("target wav 存在,已跳过: "+target_wav)
    else:
        transform.build(source_wav, target_wav, return_output=False)
    duration = float(subprocess.check_output('soxi -D {0}'.format(target_wav), shell=True))
    processed_info.append({"audio_filepath": target_wav, "duration": duration, "text": text})

def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers
    if data_root is None:
        data_root = './data'  # 数据集默认储存路径
    if data_sets == "ALL":
        data_sets = "dev-clean,dev-other,train-clean-100,train-clean-360,train-other-500,test-clean,test-other"
    if data_sets == "BASE":  # 只有包含dev-clean train100 数据集用于快速验证
        data_sets = "dev-clean,train-clean-100"

    for data_set in data_sets.split(','):
        logging.info("\n\nWorking on: {0}".format(data_set))
        filepath = os.path.join(data_root, data_set + ".tar.gz")
        logging.info("Getting {0}".format(data_set))
        __maybe_download_file(filepath, data_set.upper())
        logging.info("Extracting {0}".format(data_set))
        __extract_file(filepath, data_root)
        logging.info("Processing {0}".format(data_set))
        __process_data(
            os.path.join(os.path.join(data_root, "LibriSpeech"), data_set.replace("_", "-"),),
            os.path.join(os.path.join(data_root, "LibriSpeech"), data_set.replace("_", "-"),) + "-processed",
            os.path.join(data_root, data_set + ".json"),
            num_workers=num_workers
        )
    logging.info('Done!')


if __name__ == "__main__":
    # python scripts/get_libri.py --data_root=/mnt/workspace2/datasets/libri/ --num_workers=12 --data_sets=ALL
    main()
