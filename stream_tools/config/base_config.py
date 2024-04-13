import logging
import pandas as pd
import requests as re
import json


class BaseConfig():
    def __init__(self, cfg):
        pass
    
    @property
    def cam_ids(self):
        return self.cams.cam_name.values
        
            
            