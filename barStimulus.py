import numpy as np
import skimage.transform as skiT

from . import Stimulus


class barStimulus(Stimulus):
    """Define a Stimulus of eight bars crossing though the FoV from different dicetions"""

    def __init__(
        self,
        stimSize=101,
        maxEcc=7,
        overlap=1 / 2,
        nBars=1,
        doubleBarRot=0,
        thickRatio=1,
        continuous=False,
        TR=2,
        stim_duration=336,
        blank_duration=12,
        loadImages=None,
        flickerFrequency=8,
        forceBarWidth=None,
        continuous_multiplier=2,
        startingDirection=[0, 3, 6, 1, 4, 7, 2, 5],
        background=128,
    ):
        # Initialize common parameters from parent
        super().__init__(
            stimSize,
            maxEcc,
            TR=TR,
            stim_duration=stim_duration,
            blank_duration=blank_duration,
            loadImages=loadImages,
            flickerFrequency=flickerFrequency,
            continuous=continuous,
            background=background,
        )
        self.stimulus_type = "bar"
        self.startingDirection = startingDirection
        self.nBars = nBars
        self.doubleBarRot = doubleBarRot
        self.thickRatio = thickRatio
        self.forceBarWidth = forceBarWidth

        self.crossings = len(self.startingDirection)
        self.nBlanks = 4

        # Compute frames per crossing and bar width parameters
        self.framesPerCrossing = self._compute_frames_per_crossing(
            continuous, blank_duration
        )
        self._compute_bar_width(overlap)

        if not continuous:
            self.continuous = False
            self._init_non_continuous()
        else:
            self.continuous = True
            self.frameMultiplier = self.TR * continuous_multiplier
            self.ncontinuousFrames = int(self.nFrames * self.frameMultiplier)
            self.continuousBlankLength = int(blank_duration * self.frameMultiplier)
            # Recompute framesPerCrossing using continuous parameters
            self.framesPerCrossing = self._compute_frames_per_crossing(
                continuous, blank_duration
            )
            self._init_continuous()

    def _compute_frames_per_crossing(self, continuous, blank_duration):
        if not continuous:
            return int(
                (self.nFrames - self.nBlanks * self.blankLength) / self.crossings
            )
        else:
            return int(
                (self.ncontinuousFrames - 4 * self.continuousBlankLength)
                / self.crossings
            )

    def _compute_bar_width(self, overlap):
        if self.forceBarWidth is not None:
            forceBarWidthPix = np.ceil(
                self.forceBarWidth / (2 * self._maxEcc) * self._stimSize
            ).astype(int)
            self.bar_width = forceBarWidthPix
            self.overlap = (self._stimSize + 0.5) / (
                self.bar_width * self.framesPerCrossing
            )
        else:
            self.overlap = overlap
            self.bar_width = np.ceil(
                self._stimSize / (self.framesPerCrossing * self.overlap - 0.5)
            ).astype(int)

    def _init_non_continuous(self):
        self._stimRaw = np.zeros((self.nFrames, self._stimSize, self._stimSize))
        self._stimBase = np.zeros(
            self.nFrames
        )  # to determine which checker pattern to use
        self.jump_size = self.overlap * self.bar_width

        it = 0
        for cross in self.startingDirection:
            for i in range(self.framesPerCrossing):
                frame = np.zeros((self._stimSize, self._stimSize))
                start = max(0, int(self.jump_size * (i - 1)))
                end = min(
                    self._stimSize, int(self.jump_size * (i - 1) + self.bar_width)
                )
                frame[:, start:end] = 1

                if self.nBars > 1:
                    frame = self._apply_multiple_bars(frame, i)

                rotated_frame = skiT.rotate(
                    frame, cross * 360 / self.crossings, order=0
                )
                self._stimRaw[it, ...] = rotated_frame
                self._stimBase[it] = np.mod(cross, 2) + 1
                it += 1
            if cross % 2 != 0:
                it += self.blankLength

        self._create_mask(self._stimRaw.shape)
        self._stimUnc = np.zeros(self._stimRaw.shape)
        self._stimUnc[:, self._stimMask] = self._stimRaw[:, self._stimMask]

    def _init_continuous(self):
        self._stimRaw = np.zeros(
            (self.ncontinuousFrames, self._stimSize, self._stimSize)
        )
        self._stimBase = np.zeros(self.ncontinuousFrames)
        self.jump_size = self.overlap * self.bar_width / self.frameMultiplier

        it = 0
        for cross in self.startingDirection:
            for i in range(self.framesPerCrossing):
                frame = np.zeros((self._stimSize, self._stimSize))
                start = max(0, int(self.jump_size * (i - self.frameMultiplier)))
                end = min(
                    self._stimSize,
                    int(self.jump_size * (i - self.frameMultiplier) + self.bar_width),
                )
                frame[:, start:end] = 1
                rotated_frame = skiT.rotate(
                    frame, cross * 360 / self.crossings, order=0
                )
                self._stimRaw[it, ...] = rotated_frame
                self._stimBase[it] = np.mod(cross, 2) + 1
                it += 1
            if cross % 2 != 0:
                it += self.continuousBlankLength

        self._create_mask(self._stimRaw.shape)
        self._stimUnc = np.zeros(self._stimRaw.shape)
        self._stimUnc[:, self._stimMask] = self._stimRaw[:, self._stimMask]

    def _apply_multiple_bars(self, frame, i):
        """Helper to modify frame when multiple bars are used."""
        self.nBarShift = self._stimSize // self.nBars
        frame2 = np.zeros((self._stimSize, self._stimSize))
        frame3 = np.zeros((self._stimSize, self._stimSize))

        for nbar in range(self.nBars - 1):
            if i == 0:
                o = max(
                    int(np.ceil(self._stimSize / 2 - 1)),
                    int(
                        self.jump_size * (i - 1)
                        + self.bar_width * self.thickRatio * 0.55
                        + self.nBarShift * (nbar + 1)
                    ),
                )
                t = int(
                    self.jump_size * (i - 1)
                    + self.bar_width * self.thickRatio
                    + self.nBarShift * (nbar + 1)
                )
            elif i == self.framesPerCrossing - 1:
                o = int(self.jump_size * (i - 1) + self.nBarShift * (nbar + 1))
                t = int(
                    self.jump_size * (i - 1)
                    + self.bar_width * self.thickRatio * 0.45
                    + self.nBarShift * (nbar + 1)
                )
            else:
                o = int(self.jump_size * (i - 1) + self.nBarShift * (nbar + 1))
                t = int(
                    self.jump_size * (i - 1)
                    + self.bar_width * self.thickRatio
                    + self.nBarShift * (nbar + 1)
                )

            frame2[:, max(0, o) : min(self._stimSize, t)] = 1
            frame3[
                :,
                max(0, o - self._stimSize) : max(
                    0,
                    min(
                        int(np.floor(self._stimSize / 2)),
                        min(self._stimSize, t - self._stimSize),
                    ),
                ),
            ] = 1
            frame2 = skiT.rotate(frame2, self.doubleBarRot, order=0)
            frame3 = skiT.rotate(frame3, self.doubleBarRot, order=0)
            frame = np.any(np.stack((frame, frame2, frame3), axis=2), axis=2)
        return frame

    def _checkerboard(self, nChecks=10):
        """Create the four flickering main images for the stimulus."""
        self._compute_check_parameters(nChecks)

        # Define cropping slices for two sets of images
        crop_main = (
            int(self.checkSize * 3 / 2),
            -int(self.checkSize / 2),
            int(self.checkSize / 2),
            -int(self.checkSize * 3 / 2),
        )
        crop_rotated = (
            self.checkSize,
            -self.checkSize,
            self.checkSize,
            -self.checkSize,
        )

        # Create basic patterns using np.tile for clarity.
        patternA = np.tile([[0, 255], [255, 0]], (self.nChecks + 1, self.nChecks + 1))
        patternB = np.tile([[255, 0], [0, 255]], (self.nChecks + 1, self.nChecks + 1))

        # Generate and crop checkA and checkB
        self.checkA = self._generate_and_crop(patternA, crop_main)
        self.checkB = self._generate_and_crop(patternB, crop_main)

        # Generate rotated versions for checkC and checkD using skimage.transform.rotate
        rotatedA = skiT.rotate(
            np.kron(patternA, np.ones((self.checkSize, self.checkSize))),
            angle=45,
            resize=False,
            order=0,
        )
        rotatedB = skiT.rotate(
            np.kron(patternB, np.ones((self.checkSize, self.checkSize))),
            angle=45,
            resize=False,
            order=0,
        )
        self.checkC = np.where(rotatedA < self.background, 0, 255)[
            crop_rotated[0] : crop_rotated[1], crop_rotated[2] : crop_rotated[3]
        ]
        self.checkD = np.where(rotatedB < self.background, 0, 255)[
            crop_rotated[0] : crop_rotated[1], crop_rotated[2] : crop_rotated[3]
        ]

        # Ensure the computed check images match the stimulus size
        self.checkA = self._crop_to_stimSize(self.checkA)
        self.checkB = self._crop_to_stimSize(self.checkB)
        self.checkC = self._crop_to_stimSize(self.checkC)
        self.checkD = self._crop_to_stimSize(self.checkD)

    def _compute_check_parameters(self, nChecks):
        """Compute checkSize and nChecks based on stimulus size and bar width."""
        if hasattr(self, "bar_width"):
            self.checkSize = int(
                np.min(
                    (
                        np.ceil(self._stimSize / nChecks / 2).astype(int),
                        np.ceil(self.bar_width / 1.5),
                    )
                )
            )
            self.nChecks = int(np.ceil(self._stimSize / self.checkSize / 2))
        else:
            self.checkSize = int(np.ceil(self._stimSize / nChecks / 2))
            self.nChecks = nChecks

    def _generate_and_crop(self, pattern, crop):
        """
        Generate a checker image using np.kron and crop it.

        crop: tuple (top, bottom, left, right) used as slice indices.
        """
        board = np.kron(pattern, np.ones((self.checkSize, self.checkSize)))
        return board[crop[0] : crop[1], crop[2] : crop[3]]

    def _crop_to_stimSize(self, img):
        """
        Crop the image to match the stimulus size.

        If the image dimensions do not match self._stimSize, the image is cropped equally
        from all sides.
        """
        if img.shape[0] != self._stimSize:
            diff = (img.shape[0] - self._stimSize) / 2
            img = img[
                np.ceil(diff).astype(int) : img.shape[0] - np.floor(diff).astype(int),
                np.ceil(diff).astype(int) : img.shape[1] - np.floor(diff).astype(int),
            ]
        return img

    def stimulus_length(self):
        super().stimulus_length()
        print(f"Frames per crossing: {self.framesPerCrossing} TRs")
        print(f"blanks: {self.nBlanks} x {self.blankLength} TRs")

    def bar_params(self):
        if self.nBars > 1:
            print(f"nBars: {self.nBars}")
            print(f"shift between bars: {self.nBarShift}")
            print(f"bar rotation: {self.doubleBarRot}")
        print(
            f"Bar width: {np.float64(np.round(self.bar_width / self._stimSize * self._maxEcc * 2, 2))}°"
        )
