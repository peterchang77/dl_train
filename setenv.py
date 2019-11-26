import os, sys, subprocess 

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
    
    download_and_unzip_data(URLS[DS_NAME], DS_PATH)
    
def download_and_unzip_data(URL, DST):
    
    # --- Download
    os.makedirs(DST, exist_ok=True)
    zip_ = '%s/raw.zip' % DST
    
    args = ['wget', '-O']
    args.append(zip_)
    args.append(URL)
    
    subprocess.run(args)
    
    # --- Unzip
    args = ['unzip', '-o']
    args.append(zip_)
    args.append('-d')
    args.append(DST)

    subprocess.run(args)

def prepare_environment(DL_PATH, DS_PATH=None, DS_NAME=None, ignore_existing=False, CUDA_VISIBLE_DEVICE=0):
    
    prepare_path(DL_PATH) 
    
    if DS_PATH is not None:
        prepare_data(DS_PATH, DS_NAME, ignore_existing)

    os.environ['CUDA_VISIBLE_DEVICE'] = CUDA_VISIBLE_DEVICE
