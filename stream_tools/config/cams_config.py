import logging
import pandas as pd
import requests as re
import json

from stream_tools.config.base_config import BaseConfig


class CamsConfig(BaseConfig):
    def __init__(self, cfg):
        self.api_url = cfg.get('api_url', None)
        self.access_token = cfg.get('access_token', None)
        self.api_args = cfg.get('api_args', None)
        self.cameras = cfg.get('cameras', None)
        self.cams = {
            'cam_name': [],
            'ivideon_id': [],
            'rtsp': [],
            'address': [],
        }
        for cam_name, params in self.cameras.items():
            self.cams['cam_name'].append(cam_name)
            self.cams['ivideon_id'].append(params.get('ivideon_id', None))
            self.cams['rtsp'].append(params.get('rtsp', None))
            self.cams['address'].append(params.get('address', None))
        self.cams = pd.DataFrame(self.cams)
        self.check_sources()
    
    def check_sources(self):
        for _, row in self.cams.iterrows():
            if row['ivideon_id'] is None and row['rtsp'] is None:
                raise ValueError(f'Camera source is not specified: {row}')
    
    def update_cams(self, correct_idx):
        self.cams = self.cams.loc[correct_idx]
    
    @property
    def addresses(self):
        return self.cams.address.values
    
    @property
    def ivideon_ids(self):
        return self.cams.ivideon_id.values
    
    @property
    def sources(self):
        sources = [None for _ in range(len(self.cams))]
        for i, row in self.cams.iterrows():
            if row['rtsp'] is not None:
                sources[i] = row['rtsp']
            elif row['ivideon_id'] is not None:
                sources[i] = row['ivideon_id']
            else:
                raise ValueError(f'Camera source is not specified: {row}')
        return sources
    
    @property
    def source_types(self):
        source_types = [None for _ in range(len(self.cams))]
        for i, row in self.cams.iterrows():
            if row['rtsp'] is not None:
                source_types[i] = 'rtsp'
            elif row['ivideon_id'] is not None:
                source_types[i] = 'ivideon'
            else:
                raise ValueError(f'Camera source is not specified: {row}')
        return source_types
        

class CamsConfigPerimeter(CamsConfig):
    def __init__(self, cfg):
        super(CamsConfigPerimeter, self).__init__(cfg)
        perimeter_info = {
            'lines': [],
            'zones': [],
            'wh': []
        }
        for _, params in self.cameras.items():
            perimeter_info['lines'].append(params.get('lines', {}))
            perimeter_info['zones'].append(params.get('zones', {}))
            perimeter_info['wh'].append(params.get('wh', [1280, 720]))
        perimeter_info = pd.DataFrame(perimeter_info)
        self.cams = pd.concat([self.cams, perimeter_info], axis=1)
    
    @property
    def zones(self):
        return self.cams.zones.values
    
    @property
    def lines(self):
        return self.cams.lines.values
    
    @property
    def resolution_wh(self):
        return self.cams.wh.values