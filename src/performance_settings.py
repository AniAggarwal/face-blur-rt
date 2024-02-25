class PerformanceSettings:
    def __init__(self, mode="high_accuracy"):
        self.mode = mode

    def adjust_settings(self, mode):
        """Adjust performance settings."""
        self.mode = mode
