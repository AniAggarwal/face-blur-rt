# TODO: somehow figure out how to intergrate this with settings like
# resolution, target_fps, etc., esp. since ML models will be trained
# at a certain resolution and fps


class PerformanceSettings:
    def __init__(
        self,
        resolution: tuple[int, int],
        target_fps: int,
        fps_counter: bool = True,
        enable_recognition: bool = True,
        enable_labels: bool = True,
        apply_blur: bool = True,
        display_video: bool = True,
    ) -> None:
        self.resolution = resolution
        self.target_fps = target_fps
        self.fps_counter = fps_counter
        self.enable_recognition = enable_recognition
        self.enable_labels = enable_labels
        self.apply_blur = apply_blur
        self.display_video = display_video
