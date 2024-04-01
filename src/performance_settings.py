# TODO: somehow figure out how to intergrate this with settings like
# resolution, target_fps, etc., esp. since ML models will be trained
# at a certain resolution and fps


class PerformanceSettings:
    def __init__(self, resolution: tuple[int, int], target_fps: int) -> None:
        self.resolution = resolution
        self.target_fps = target_fps
