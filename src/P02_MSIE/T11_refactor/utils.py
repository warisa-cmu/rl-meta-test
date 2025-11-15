from dataclasses import dataclass
import numpy as np


# LinearScaler for parameter scaling
@dataclass
class LinearScaler:
    bounds: tuple[float, float]  # Original bounds
    bounds_scaled: tuple[float, float] = (0.0, 1.0)  # Scaled bounds
    invalid_value: float | None = (
        None  # Value to use when parameter is invalid (e.g., out of bounds)
    )
    starting_value: float | None = None  # Initial value for the parameter (unscaled)

    def __post_init__(self):
        pass

    def transform(self, vin, keep_list=False):
        is_list = False
        if type(vin) in [list, tuple]:
            is_list = True
            vin = np.array(vin)

        if np.any(vin < self.bounds[0]) or np.any(vin > self.bounds[1]):
            raise ValueError(
                f"Value {vin} out of bounds for scaling. bounds: {self.bounds}"
            )

        # Min-max scaling to new bounds
        old_min, old_max = self.bounds
        new_min, new_max = self.bounds_scaled
        if old_max == old_min:
            # Handle the case where the old range is a single point
            return (
                new_min
                if vin == old_min
                else ValueError("Cannot scale value outside of the old bounds.")
            )
        scaled_value = new_min + ((vin - old_min) * (new_max - new_min)) / (
            old_max - old_min
        )
        if keep_list and is_list:
            return scaled_value.tolist()
        return scaled_value

    def inverse_transform(self, vin, keep_list=False):
        is_list = False
        if type(vin) in [list, tuple]:
            is_list = True
            vin = np.array(vin)
        if np.any(vin < self.bounds_scaled[0]) or np.any(vin > self.bounds_scaled[1]):
            raise ValueError(
                f"Value {vin} out of bounds for inverse scaling. bounds_scaled: {self.bounds_scaled}"
            )

        # Inverse min-max scaling to original bounds
        old_min, old_max = self.bounds
        new_min, new_max = self.bounds_scaled
        if new_max == new_min:
            # Handle the case where the new range is a single point
            return (
                old_min
                if vin == new_min
                else ValueError("Cannot inverse scale value outside of the new bounds.")
            )
        original_value = old_min + ((vin - new_min) * (old_max - old_min)) / (
            new_max - new_min
        )
        if keep_list and is_list:
            return original_value.tolist()
        return original_value

    def clip_to_bounds(self, value, scaled=False):
        if scaled:
            return np.clip(value, self.bounds_scaled[0], self.bounds_scaled[1])
        return np.clip(value, self.bounds[0], self.bounds[1])

    def get_invalid_value(self, scaled=False):
        if self.invalid_value is None:
            raise ValueError("Invalid value is not set for this scaler.")
        if scaled:
            return self.transform(self.invalid_value)
        return self.invalid_value

    def get_starting_value(self, scaled=False):
        if self.starting_value is None:
            raise ValueError("Starting value is not set for this scaler.")
        if scaled:
            return self.transform(self.starting_value)
        return self.starting_value
