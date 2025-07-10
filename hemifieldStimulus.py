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
        stim_duration=300,
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
            stim_duration=stim_duration,
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
        self.total_duration = stim_duration
        self.whichCheck = whichCheck
        self._loadImages = loadImages
        self._carrier = "images" if self._loadImages is not None else "checker"

        self.nTRs = int(np.ceil(self.total_duration / self.TR))

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
        """Generate a sequence: 1 (on), 2 (on), 0 (off) with random durations, repeat until run time-20s, then shuffle blocks so that no same conditions are adjacent."""
        import random

        blocks = []
        current_time = 0
        # Generate blocks until we reach total_duration - 20s
        while current_time < self.total_duration - 20:
            # Condition 1 (on)
            dur1 = self.stim_on_duration + random.uniform(
                -self.stim_jitter, self.stim_jitter
            )
            dur1 = max(self.TR, dur1)
            blocks.append((1, dur1))
            current_time += dur1
            if current_time >= self.total_duration - 20:
                break
            # Condition 2 (on)
            dur2 = self.stim_on_duration + random.uniform(
                -self.stim_jitter, self.stim_jitter
            )
            dur2 = max(self.TR, dur2)
            blocks.append((2, dur2))
            current_time += dur2
            if current_time >= self.total_duration - 20:
                break
            # Condition 0 (off)
            duroff = self.stim_off_duration + random.uniform(
                -self.off_jitter, self.off_jitter
            )
            duroff = max(self.TR, duroff)
            blocks.append((0, duroff))
            current_time += duroff

        # Shuffle blocks, keep the shuffle with the least adjacent repeats
        def count_adjacent_repeats(lst):
            return sum(lst[i][0] == lst[i + 1][0] for i in range(len(lst) - 1))

        best_blocks = None
        min_repeats = float("inf")
        for _ in range(1000):
            random.shuffle(blocks)
            repeats = count_adjacent_repeats(blocks)
            if repeats < min_repeats:
                min_repeats = repeats
                best_blocks = list(blocks)
            if repeats == 0:
                break

        # Use the best shuffle found
        blocks = best_blocks

        # Convert blocks to TR-wise sequence
        seq = []
        for cond, dur in blocks:
            ntr = int(round(dur / self.TR))
            seq.extend([cond] * ntr)

        # If too short, pad with off; if too long, trim
        nTRs = self.nTRs
        if len(seq) < nTRs:
            seq.extend([0] * (nTRs - len(seq)))
        elif len(seq) > nTRs:
            seq = seq[:nTRs]

        # Store as periods for compatibility
        self._sequence = []
        start_tr = 0
        while start_tr < nTRs:
            cond = seq[start_tr]
            end_tr = start_tr + 1
            while end_tr < nTRs and seq[end_tr] == cond:
                end_tr += 1
            self._sequence.append(
                {
                    "condition": cond,
                    "start_tr": start_tr,
                    "duration_tr": end_tr - start_tr,
                    "end_tr": end_tr,
                }
            )
            start_tr = end_tr

    def _create_hemisphere_stimulus(self):
        """Create the flickering hemisphere stimulus, calculated in TRs."""

        self._stimUnc = np.ones((self.nTRs, self._stimSize, self._stimSize))

        # Create checkerboard patterns if no images provided
        if self._loadImages is None:
            self._checkerboard()
        else:
            self._loadCarrierImages(self._loadImages)

        # Fill in the stimulus according to sequence
        for period in self._sequence:
            start_tr = period["start_tr"]
            end_tr = period["end_tr"]
            condition = period["condition"]

            if condition == 0:
                continue
            elif condition == 1:
                mask = self._left_mask
            elif condition == 2:
                mask = self._right_mask

            for tr_idx in range(start_tr, end_tr):
                if tr_idx >= self.nTRs:
                    break
                self._stimUnc[tr_idx] = mask

        # Store sequence info for analysis
        self._onsets = [p["start_tr"] * self.TR for p in self._sequence]
        self._conditions = [p["condition"] for p in self._sequence]
        self._durations = [p["duration_tr"] * self.TR for p in self._sequence]

        self._stimBase = np.ones(self._stimUnc.shape[0])

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
