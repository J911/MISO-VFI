from .dataloader_vimeo import load_data as load_vimeo
from .dataloader_vimeo_triplet import load_data as load_vimeo_tri

def load_data(dataname,batch_size, test_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'vimeo':
        return load_vimeo(batch_size, test_batch_size, data_root, num_workers)
    if dataname == 'vimeo-triplet':
        return load_vimeo_tri(batch_size, test_batch_size, data_root, num_workers)
