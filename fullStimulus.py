import numpy as np
import skimage.transform as skiT

from .barStimulus import barStimulus
from .stimulus import Stimulus
from .wedgeStimulus import wedgeStimulus


class fullStimulus(Stimulus):
    """Define a Stimulus of fullfield flickering"""

    def __init__(
        self,
        stimSize=101,
        maxEcc=7,
        TR=2,
        stim_duration=336,
        off_duration=12,
        on_duration=2,
        nTrials=None,
        loadImages=None,
        flickerFrequency=8,
        whichCheck="bar",
        jitter=False,
    ):
        self.stimulus_type = "full"

        self._maxEcc = maxEcc
        self._stimSize = stimSize
        self.TR = TR
        self._loadImages = loadImages
        self._carrier = "images" if self._loadImages is not None else "checker"
        self.continuous = np.mod(on_duration, TR) > 0

        if nTrials is not None:
            self.nTrials = nTrials
        else:
            self.nTrials = int(stim_duration / (on_duration + off_duration))

        if isinstance(on_duration, int):
            self.on_duration = [on_duration] * self.nTrials
        elif isinstance(on_duration, list):
            if len(on_duration) == self.nTrials:
                self.on_duration = np.array(on_duration).astype(float)
            else:
                self.on_duration = np.random.choice(on_duration, self.nTrials)
                if len(on_duration) <= self.nTrials:
                    while not np.all([i in self.on_duration for i in on_duration]):
                        self.on_duration = np.random.choice(on_duration, self.nTrials)
            self.continuous = True

        if isinstance(off_duration, int):
            self.off_duration = [off_duration] * self.nTrials
        elif isinstance(off_duration, list):
            if len(off_duration) == self.nTrials:
                self.off_duration = np.array(off_duration).astype(float)
            else:
                self.off_duration = np.random.choice(off_duration, self.nTrials)
                if len(off_duration) <= self.nTrials:
                    while not np.all([i in self.off_duration for i in off_duration]):
                        self.off_duration = np.random.choice(off_duration, self.nTrials)
            self.continuous = True

        self.stim_duration = int(sum(self.off_duration) + sum(self.on_duration))

        self.nFrames = np.ceil(self.stim_duration / self.TR).astype(int)

        self.flickerFrequency = flickerFrequency  # Hz

        self.whichCheck = whichCheck
        self.crossings = self.nTrials
        if jitter:
            self.continuous = True
            if isinstance(jitter, int):
                self.jitter = jitter
            else:
                self.jitter = 3
        else:
            self.jitter = False

        self.min_blank = 10  # s

        if not self.continuous:
            n_on_frames = np.round(self.on_duration / self.TR).astype(int)
            n_off_frames = np.round(self.off_duration / self.TR).astype(int)

            # self.blankLength = np.ceil(off_duration / self.TR).astype(int)
            # off_frames = [np.zeros((self.blankLength, self._stimSize, self._stimSize))] * self.nTrials

            # ff = self.nFrames

        else:
            self.frameMultiplier = self.TR * self.flickerFrequency / 2
            # self.continuousBlankLength = int(off_duration / self.TR * self.frameMultiplier)

            # if not self.jitter:
            #     off_frames = [np.zeros((self.continuousBlankLength, self._stimSize, self._stimSize))] * self.nTrials
            # else:
            #     min_blank_cont = np.ceil(self.min_blank / self.TR * self.frameMultiplier).astype(int)
            #     jj = np.ceil(self.jitter / self.TR  * self.frameMultiplier).astype(int)
            #     blank_length_frames = np.round(np.random.uniform(max(min_blank_cont, self.continuousBlankLength - jj),
            #                                   self.continuousBlankLength + jj,
            #                                   self.nTrials)).astype(int)

            n_on_frames = np.round(
                self.on_duration / self.TR * self.frameMultiplier
            ).astype(int)
            n_off_frames = np.round(
                self.off_duration / self.TR * self.frameMultiplier
            ).astype(int)

        on_frames = [np.ones((i, self._stimSize, self._stimSize)) for i in n_on_frames]
        off_frames = [
            np.zeros((i, self._stimSize, self._stimSize)) for i in n_off_frames
        ]

        self._stimRaw = np.zeros((0, self._stimSize, self._stimSize))
        self._onsets = np.zeros((self.nTrials * 2))
        self._onsets[0] = 0
        for I, (on_frame, off_frame) in enumerate(zip(on_frames, off_frames)):
            self._stimRaw = np.concatenate((self._stimRaw, on_frame, off_frame), 0)

            self._onsets[2 * I + 1] = self._onsets[2 * I] + len(on_frame)
            if I < self.nTrials - 1:
                self._onsets[2 * I + 2] = self._onsets[2 * I + 1] + len(off_frame)

        if self.continuous:
            self._onsets = self._onsets * 2 / self.flickerFrequency
            ff = len(self._stimRaw)
            self.ncontinuousFrames = ff

        self._stimBase = np.ones(ff)  # to find which checkerboard to use

        if len(self._stimRaw) < ff:
            diff = ff - len(self._stimRaw)
            self._stimRaw = np.concatenate(
                (self._stimRaw, np.zeros((diff, self._stimSize, self._stimSize))), 0
            )

        self._create_mask(self._stimRaw.shape)

        self._stimUnc = np.zeros(self._stimRaw.shape)
        self._stimUnc[:, self._stimMask] = self._stimRaw[:, self._stimMask]

    def _checkerboard(self):
        if "bar" in self.whichCheck:
            self.barWidth = 1e6
            barStimulus._checkerboard(self, nChecks=10)
        elif "wedge" in self.whichCheck:
            wedgeStimulus._checkerboard(self, nFlickerRings=18, nFlickerWedge=24)
        else:
            print(f'Please choose whichCheck form ["bar", "wedge"]!')


if __name__ == "__main__":
    from PRFstimulus import fullStimulus

    foo = fullStimulus(
        on_duration=2, TR=1.75, maxEcc=9, nTrials=10, off_duration=25, jitter=4
    )
    foo.playVid(flicker=True)
