import os, sys, subprocess 
from zipfile import ZipFile

def prepare_path(DL_PATH):
    
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
    
    # --- Set DL_CLIENT_SUMMARY_PATH
    os.environ['DL_CLIENT_SUMMARY_PATH'] = '%s/pkls/summary.pkl' % DS_PATH
    os.environ['DS_PATH'] = DS_PATH
    
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
            print('\rDownloading %s dataset to %s: %05.3f MB / %05.3f MB' % (DS_NAME, dst, dload_size / 1e6, total_size / 1e6), end=' ', flush=True)
            f.write(data)
            
def unzip_data(fname, dst):
    
    zf = ZipFile(fname)

    fnames = zf.infolist()
    total_size = sum([f.file_size for f in fnames])
    unzip_size = 0

    for f in fnames:
        
        unzip_size += f.file_size
        print('\rExtracting zip archive: %05.3f MB / %05.3f MB' % (unzip_size / 1e6, total_size / 1e6), end=' ', flush=True)
        zf.extract(f, '%s/%s' % (dst, f.filename))

def prepare_environment(DL_PATH, DS_PATH=None, DS_NAME=None, ignore_existing=False, CUDA_VISIBLE_DEVICES=0):
    
    prepare_path(DL_PATH) 
    
    if DS_PATH is not None:
        prepare_data(DS_PATH, DS_NAME, ignore_existing)

    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
