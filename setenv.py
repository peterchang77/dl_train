import os, sys, requests, subprocess, time
from zipfile import ZipFile

def prepare_path(DL_PATH):

    # --- Check default locations
    if DL_PATH is None:

        if os.getcwd().split('/')[-4] == 'dl_core':
            DL_PATH = '../../'

        elif glob.glob('/home/*/python/dl_core') > 0:
            DL_PATH = sorted(glob.glob('/home/*/python/dl_core'))[0]

        else:
            DL_PATH = '%s/python/dl_core' % os.environ['HOME']
    
    # --- Pull rep
    if not os.path.exists(DL_PATH):
        args = ['git', 'clone', 'https://github.com/peterchang77/dl_core', DL_PATH]
        subprocess.run(args)
        
    # --- Update repo
    else:
        args = ['git', '-c', DL_PATH, 'pull', 'origin', 'master']
        subprocess.run(args)
        
    if DL_PATH not in sys.path:
        sys.path.append(DL_PATH)

def prepare_data(DS_PATH, DS_NAME, ignore_existing=False):
    
    # --- Set DS_PATH 
    os.environ['DS_PATH'] = DS_PATH or '/data/raw/%s' % DS_NAME
    
    if os.path.exists(DS_PATH) and not ignore_existing:
        return
    
    URLS = {
        'brats': 'https://www.dropbox.com/s/wuady574manrwew/brats.zip?dl=1'}

    assert DS_NAME in URLS, 'ERROR provided dataset name is not recognized'
    
    download_and_unzip_data(URLS[DS_NAME], DS_PATH, DS_NAME)
    
def download_and_unzip_data(URL, DST, DS_NAME):
    
    # --- Download
    zip_ = '%s/raw.zip' % DST
    os.makedirs(DST, exist_ok=True)
    download_data(URL, zip_, DS_NAME)
    
    # --- Unzip
    unzip_data(zip_, DST)
    
def download_data(url, dst, DS_NAME):
    
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 32768
    dload_size = 0

    with open(dst, 'wb') as f:
        for data in r.iter_content(block_size):
            dload_size += len(data)
            printp('Downloading %s dataset to %s (%05.3f MB / %05.3f MB)' % 
                (DS_NAME, dst, dload_size / 1e6, total_size / 1e6), dload_size / total_size)
            f.write(data)

    printd('Completed download of %s dataset to %s (%05.3f MB / %05.3f MB)' % 
        (DS_NAME, dst, dload_size / 1e6, total_size / 1e6))
            
def unzip_data(fname, dst):
    
    zf = ZipFile(fname)

    fnames = zf.infolist()
    total_size = sum([f.file_size for f in fnames])
    unzip_size = 0

    for f in fnames:
        if f.filename[-1] != '/':
            unzip_size += f.file_size
            printp('Extracting zip archive (%05.3f MB / %05.3f MB)' % 
                (unzip_size / 1e6, total_size / 1e6), unzip_size / total_size)
            zf.extract(f, dst)

    printd('Completed extraction (%05.3f MB / %05.3f MB)' % 
            (unzip_size / 1e6, total_size / 1e6))

def printd(s, ljust=120, flush=False):

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    s = '\r[ %s ] %s' % (t, s)
    print(s.ljust(ljust), flush=flush)

def printp(s, progress, pattern='%0.3f', SIZE=20, ljust=120, flush=False):

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) 
    a = ''.join(['='] * int(SIZE * progress))
    b = ''.join(['.'] * (SIZE - len(a) - 1))
    c = '>' if progress < 1 else ''

    pattern = '\r[ %s ] [%s%s%s] ' + pattern + '%% : %s'
    s = pattern % (t, a, c, b, progress * 100, s)
    print(s.ljust(ljust), flush=flush, end=' ')

def prepare_environment(DL_PATH=None, DS_PATH=None, DS_NAME=None, ignore_existing=False, CUDA_VISIBLE_DEVICES=0):
    
    prepare_path(DL_PATH) 
    prepare_data(DS_PATH, DS_NAME, ignore_existing)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
