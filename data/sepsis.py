import os
import urllib
import zipfile


def download(save_path):

    base_loc = os.path.join(save_path,'sepsis')
    loc_Azip = os.path.join(base_loc, 'training_setA.zip')
    loc_Bzip = os.path.join(base_loc, 'training_setB.zip')
    
    if not os.path.exists(loc_Azip):
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                   str(loc_Azip))
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                   str(loc_Bzip))

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(str(base_loc))
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(str(base_loc))
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(os.path.join(base_loc,folder)):
                if os.path.exists(os.path.join(base_loc,filename)):
                    raise RuntimeError
                os.rename(os.path.join(base_loc, folder, filename), os.path.join(base_loc, filename))
