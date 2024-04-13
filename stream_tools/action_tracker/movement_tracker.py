from datetime import datetime

class MovementTracker:
    def __init__(self, 
                 object_id, 
                 timestamp,
                 movement_threshold=5, 
                 log_file='movement_log.txt',
                 disappear_time=10):
        self.object_id = object_id
        self.movement_threshold = movement_threshold
        self.log_file = log_file
        self.last_position = None
        self.last_time_moved = timestamp
        self.last_time = None
        self.is_moving = False
        self.last_time_stationary = timestamp
        self.disappear_time = None

    def update_position(self, position, timestamp):
        moving = True
        if self.last_position is not None:
            distance = self.calculate_distance(self.last_position, position)
            if distance > self.movement_threshold:
                self.log_event(
                    f"{self.object_id} moving {self.format_time(self.last_time)} : {self.format_time(timestamp)}"
                )
            else:
                self.log_event(
                    f"{self.object_id} stationary {self.format_time(self.last_time)} : {self.format_time(timestamp)}"
                )
                moving = False
        self.last_position = position
        self.last_time = timestamp
        return moving
    
    # def update_position(self, position, timestamp):
    #     if self.last_position is not None:
    #         distance = self.calculate_distance(self.last_position, position)
    #         if distance > self.movement_threshold:
    #             if not self.is_moving:
    #                 self.is_moving = True
    #                 self.last_time_moved = timestamp
    #                 if self.last_time_stationary:
    #                     self.log_event(
    #                         f"{self.object_id} stationary {self.format_time(self.last_time_stationary)} : {self.format_time(timestamp)}"
    #                         )
    #             self.last_position = position
    #             self.log_event(f"f{self.object_id} moving {self.format_time(self.last_time)} : {self.format_time(timestamp)}")
    #         else:
    #             if self.is_moving:
    #                 self.is_moving = False
    #                 self.last_time_stationary = timestamp
    #                 self.log_event(f"{self.object_id} started moving at {self.format_time(self.last_time_moved)}")
    #     else:
    #         self.last_position = position
    #     self.last_time = timestamp
    #     return self.is_moving

    def log_event(self, message):
        with open(self.log_file, 'a') as file:
            file.write(f"{message}\n")

    @staticmethod
    def calculate_distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    @staticmethod
    def format_time(timestamp):
        return timestamp.isoformat("T", "milliseconds").replace(
            ":", "_"
        ).replace('.', '_')

class MultiCameraMovementTracker:
    def __init__(self, movement_threshold=5, log_file='movement_log.txt'):
        self.movement_threshold = movement_threshold
        self.log_file = log_file
        self.trackers = {}

    def update_position(self, camera_id, object_id, object_class, position, timestamp):
        unique_id = (camera_id, object_id, object_class)
        if unique_id not in self.trackers:
            self.trackers[unique_id] = MovementTracker(
                object_id=unique_id, 
                timestamp=timestamp,
                movement_threshold=self.movement_threshold, 
                log_file=self.log_file)
        return self.trackers[unique_id].update_position(
            position,
            timestamp)
