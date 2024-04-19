import pandas as pd

from stream_tools.config.base_config import BaseConfig


class IvideonConfig(BaseConfig):

    def __init__(self, cfg):
        self.api_url = cfg['api_url']
        self.access_token = cfg['access_token']
        self.api_args = cfg['api_args']
        cameras = cfg['cameras']
        self.cams = {
            'cam_name': [],
            'ivideon_id': [],
            'address': [],
            'link': [], }
        for cam_name, params in cameras.items():
            self.cams['cam_name'].append(cam_name)
            self.cams['ivideon_id'].append(params['ivideon_id'])
            self.cams['address'].append(params['address'])
            self.cams['link'].append('')
        self.cams = pd.DataFrame(self.cams)

    def update_cams(self, correct_idx):
        self.cams = self.cams.loc[correct_idx]

    @property
    def links(self):
        return self.cams['ivideon_id'].values

    @property
    def ivideon_ids(self):
        return self.cams.ivideon_id.values
