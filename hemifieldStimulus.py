from copy import deepcopy

import numpy as np

from . import Stimulus
from .barStimulus import barStimulus
from .wedgeStimulus import wedgeStimulus


class hemifieldStimulus(Stimulus):
    """Flickering stimulus for left/right visual field hemispheres with randomized presentation."""

    def __init__(
        self,
        stimSize=101,
        maxEcc=10,
        TR=2,
        total_duration=300,
        stim_on_duration=7,
        stim_off_duration=10,
        stim_jitter=2,
        off_jitter=3,
        gap_degrees=15,
        loadImages=None,
        flickerFrequency=8,
        background=128,
        whichCheck="wedge",
    ):
        super().__init__(
            stimSize=stimSize,
            maxEcc=maxEcc,
            TR=TR,
            stim_duration=total_duration,
            blank_duration=0,
            loadImages=loadImages,
            flickerFrequency=flickerFrequency,
            continuous=False,
            background=background,
        )

        self.stimulus_type = "hemisphere"
        self.stim_on_duration = stim_on_duration
        self.stim_off_duration = stim_off_duration
        self.stim_jitter = stim_jitter
        self.off_jitter = off_jitter
        self.gap_degrees = gap_degrees
        self.total_duration = total_duration
        self.whichCheck = whichCheck
        self._loadImages = loadImages
        self._carrier = "images" if self._loadImages is not None else "checker"

        # Create hemisphere masks
        self._create_hemisphere_masks()

        # Generate randomized stimulus sequence
        self._generate_stimulus_sequence()

        # Create the stimulus
        self._create_hemisphere_stimulus()

    def _create_hemisphere_masks(self):
        """Create masks for left and right hemispheres with wedge-shaped gap."""
        center = self._stimSize // 2
        y, x = np.ogrid[: self._stimSize, : self._stimSize]

        # Create circular mask for field of view
        distances = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        self._fov_mask = distances <= (self._stimSize // 2)

        # Calculate angles from center (in radians)
        # Note: atan2 gives angles from -π to π, with 0 pointing right
        angles = np.arctan2(y - center, x - center)

        # Convert gap from degrees to radians
        gap_radians = np.radians(self.gap_degrees)

        # Create hemisphere masks with wedge gaps
        self._left_mask = np.zeros((self._stimSize, self._stimSize), dtype=bool)
        self._right_mask = np.zeros((self._stimSize, self._stimSize), dtype=bool)

        # Left hemisphere: angles between π/2 + gap_radians and 3π/2 - gap_radians
        # This excludes the wedge around the vertical meridians
        left_condition = (
            (angles >= (np.pi / 2 + gap_radians))
            & (angles <= (3 * np.pi / 2 - gap_radians))
        ) | (
            (angles >= (-np.pi / 2 + gap_radians))
            & (angles <= (np.pi / 2 - gap_radians))
        )
        self._left_mask = left_condition & self._fov_mask

        # Right hemisphere: angles between -π/2 + half_gap and π/2 - half_gap
        # This excludes the wedge around the vertical meridians
        right_condition = (
            (angles >= (-np.pi / 2 + gap_radians))
            & (angles <= (np.pi / 2 - gap_radians))
        ) | (
            (angles >= (np.pi / 2 + gap_radians))
            & (angles <= (3 * np.pi / 2 - gap_radians))
        )
        self._right_mask = right_condition & self._fov_mask

        # Actually, let me correct this - we want left/right visual fields:
        # Left visual field: angles pointing left (around ±π)
        # Right visual field: angles pointing right (around 0)

        # Reset masks
        self._left_mask = np.zeros((self._stimSize, self._stimSize), dtype=bool)
        self._right_mask = np.zeros((self._stimSize, self._stimSize), dtype=bool)

        # Left visual field: exclude wedge around vertical meridian (±π/2)
        left_condition = (angles >= (np.pi / 2 + gap_radians)) | (
            angles <= (-np.pi / 2 - gap_radians)
        )
        self._left_mask = left_condition & self._fov_mask

        # Right visual field: exclude wedge around vertical meridian (±π/2)
        right_condition = (angles >= (-np.pi / 2 + gap_radians)) & (
            angles <= (np.pi / 2 - gap_radians)
        )
        self._right_mask = right_condition & self._fov_mask

    def _generate_stimulus_sequence(self):
        """Generate randomized sequence of left/right/off periods with no consecutive repeats."""
        current_time = 0
        self._sequence = []
        last_condition = None  # Track the last condition to avoid repeats

        while current_time < self.total_duration:
            # Choose condition ensuring it's different from the last one
            if last_condition is None:
                # First condition can be anything
                condition = np.random.choice([0, 1, 2])
            else:
                # Choose from the two conditions that are different from last_condition
                available_conditions = [c for c in [0, 1, 2] if c != last_condition]
                condition = np.random.choice(available_conditions)

            if condition == 0:  # Off period
                duration = self.stim_off_duration + np.random.uniform(
                    -self.off_jitter, self.off_jitter
                )
                duration = max(0.1, duration)  # Minimum duration
            else:  # On period (left or right)
                duration = self.stim_on_duration + np.random.uniform(
                    -self.stim_jitter, self.stim_jitter
                )
                duration = max(0.1, duration)  # Minimum duration

            # Check if we have time for this period
            if current_time + duration > self.total_duration:
                duration = self.total_duration - current_time

            # Calculate frame indices
            start_frame = int(current_time * self.flickerFrequency)
            end_frame = int((current_time + duration) * self.flickerFrequency)

            self._sequence.append(
                {
                    "condition": condition,  # 0=off, 1=left, 2=right
                    "start_time": current_time,
                    "duration": duration,
                    "end_time": current_time + duration,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )

            # Update tracking variables
            last_condition = condition
            current_time += duration

            if current_time >= self.total_duration:
                break

    def _create_hemisphere_stimulus(self):
        """Create the flickering hemisphere stimulus."""
        total_frames = int(self.total_duration * self.flickerFrequency)
        self._stimUnc = (
            np.ones((total_frames, self._stimSize, self._stimSize)) * self.background
        )

        # Create checkerboard patterns if no images provided
        if self._loadImages is None:
            self._checkerboard()
        else:
            self._loadCarrierImages(self._loadImages)

        # Fill in the stimulus according to sequence
        for period in self._sequence:
            start_frame = period["start_frame"]
            end_frame = min(period["end_frame"], total_frames)
            condition = period["condition"]

            if condition == 0:  # Off - just background (already set above)
                continue
            elif condition == 1:  # Left hemisphere
                mask = self._left_mask
            elif condition == 2:  # Right hemisphere
                mask = self._right_mask

            # Apply flickering pattern to the masked region for ALL frames in this period
            for frame_idx in range(start_frame, end_frame):
                if frame_idx >= total_frames:
                    break

                if self._loadImages is None:
                    # Alternate between checker patterns based on frame number
                    if frame_idx % 2 == 0:
                        pattern = self.checkA
                    else:
                        pattern = self.checkB
                    # Apply pattern only to the hemisphere mask
                    self._stimUnc[frame_idx][mask] = pattern[mask]
                else:
                    # Use random image
                    img_idx = np.random.randint(self.carrierImages.shape[-1])
                    self._stimUnc[frame_idx][mask] = self.carrierImages[mask, img_idx]

        # Store sequence info for analysis
        self._onsets = [p["start_time"] for p in self._sequence]
        self._conditions = [p["condition"] for p in self._sequence]
        self._durations = [p["duration"] for p in self._sequence]

        self._stimBase = np.ones(
            self._stimUnc.shape[0]
        )  # to find which checkerboard to use

    def _checkerboard(self):
        if "bar" in self.whichCheck:
            self.barWidth = 1e6
            barStimulus._checkerboard(self, nChecks=10)
        elif "wedge" in self.whichCheck:
            wedgeStimulus._checkerboard(self, nFlickerRings=18, nFlickerWedge=24)
        else:
            print(f'Please choose whichCheck form ["bar", "wedge"]!')

    def get_stimulus_info(self):
        """Return information about the stimulus sequence."""
        return {
            "sequence": self._sequence,
            "onsets": self._onsets,
            "conditions": self._conditions,  # 0=off, 1=left, 2=right
            "durations": self._durations,
            "condition_names": ["off", "left", "right"],
        }

    def save_sequence_info(self, filename):
        """Save sequence information to a file."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "onset": self._onsets,
                "duration": self._durations,
                "condition": self._conditions,
                "condition_name": [
                    "off" if c == 0 else "left" if c == 1 else "right"
                    for c in self._conditions
                ],
            }
        )

        df.to_csv(filename, index=False)
        print(f"Sequence info saved to {filename}")
