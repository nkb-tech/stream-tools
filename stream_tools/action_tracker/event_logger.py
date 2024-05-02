from datetime import datetime
import pandas as pd

class EventLogger:
    def __init__(self, 
                 log_file='event_log.csv'):
        self.log_file = log_file
        start_df = pd.DataFrame(columns=['camera', 'timestamp', 'event', 'image'])
        start_df.to_csv(log_file, index=False)

    def log_event(self,
                  camera,
                  timestamp,
                  event,
                  image):
        data = pd.DataFrame(
            {
                'camera': [camera],
                'timestamp': [self.format_time(timestamp)],
                'event': [event],
                'image': [image]
            }
        )
        data.to_csv(self.log_file, mode='a', index=False, header=False)

    @staticmethod
    def format_time(timestamp):
        return timestamp.isoformat("T", "milliseconds").replace(
            ":", "_"
        ).replace('.', '_')
