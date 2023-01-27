import numpy as np
import skimage.transform as skiT

from .stimulus import Stimulus
from .barStimulus import barStimulus
from .wedgeStimulus import wedgeStimulus

class fullStimulus(Stimulus):
    """Define a Stimulus of fullfield flickering"""

    def __init__(
        self,
        stimSize=101,
        maxEcc=7,
        TR=2,
        stim_duration=336,
        blank_duration=12,
        on_duration=2,
        nTrials=None,
        loadImages=None,
        flickerFrequency=8,
        whichCheck='bar',
    ):

        if nTrials is not None:
            self.nTrials = nTrials
            stim_duration = int(nTrials * (on_duration + blank_duration))
        else:
            self.nTrials = int(stim_duration / (on_duration + blank_duration))
            stim_duration = int(self.nTrials * (on_duration + blank_duration))

        continous = np.mod(on_duration, TR) > 0
        self.on_duration = on_duration

        super().__init__(
            stimSize,
            maxEcc,
            TR=TR,
            stim_duration=stim_duration,
            blank_duration=blank_duration,
            loadImages=loadImages,
            flickerFrequency=flickerFrequency,
            continous=continous,
        )

        self.whichCheck = whichCheck
        self.crossings  = self.nTrials

        if not self.continous:
            nOnFrames = int(self.on_duration / self.TR)

            offFrames = np.zeros((self.blankLength, self._stimSize, self._stimSize))

            ff = self.nFrames

        else:
            self.frameMultiplier = self.TR * self.flickerFrequency / 2
            self.nContinousFrames     = int(self.nFrames     * self.frameMultiplier)
            self.continousBlankLength = int(self.blankLength * self.frameMultiplier)

            nOnFrames = int(self.on_duration / self.TR * self.frameMultiplier)

            offFrames = np.zeros((self.continousBlankLength, self._stimSize, self._stimSize))

            ff = self.nContinousFrames

        onFrames  = np.ones( (nOnFrames, self._stimSize, self._stimSize))

        self._stimRaw = np.tile(np.concatenate((onFrames, offFrames), 0), [self.nTrials,1,1])

        self._stimBase = np.ones(ff)  # to find which checkerboard to use

        if len(self._stimRaw) < ff:
            diff = ff - len(self._stimRaw)
            self._stimRaw = np.concatenate((self._stimRaw, np.zeros((diff, self._stimSize, self._stimSize))), 0)

        self._create_mask(self._stimRaw.shape)

        self._stimUnc = np.zeros(self._stimRaw.shape)
        self._stimUnc[:, self._stimMask] = self._stimRaw[:, self._stimMask]


    def _checkerboard(self):
        if 'bar' in self.whichCheck:
            barStimulus._checkerboard(self, nChecks=10)
        elif 'wedge' in self.whichCheck:
            wedgeStimulus._checkerboard(self, nFlickerRings=18, nFlickerWedge=24)
        else:
            print(f'Please choose whichCheck form ["bar", "wedge"]!')


# if __name__ == '__main__':
#     foo = fullStimulus(on_duration=.2)
#     foo.playVid(flicker=True)



