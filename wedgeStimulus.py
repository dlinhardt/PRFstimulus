import numpy as np
import skimage.transform as skiT

from . import Stimulus


class wedgeStimulus(Stimulus):
    """Define a Stimulus of roataing wedge and expanding/contracting rings"""

    def __init__(
        self,
        stimSize=101,
        maxEcc=7,
        overlap=1 / 2,
        TR=2,
        nStims=1,
        stim_duration=336,
        blank_duration=12,
    ):

        super().__init__(
            stimSize,
            maxEcc,
            TR=TR,
            stim_duration=stim_duration,
            blank_duration=blank_duration,
        )
        self.stimulus_type = 'wedge'

        self.overlap = overlap

        self.framesPerParadigm = 36
        self.wedgeWidth = 2 * np.pi * 2 / self.framesPerParadigm / self.overlap
        # self.ringWidth  =

        self.paradigms = [
            "wCCW",
            "r",
            "wCW",
            "r",
        ]  # 'wedgeCCW', 'ring','wedgeCW', 'ring'
        self._stimBase = np.ones(self.nFrames)

        # define meshgrid in polar corrdinates
        self.X, self.Y = np.meshgrid(
            np.linspace(-self._stimSize // 2, self._stimSize // 2 - 1, self._stimSize) +
            0.5,
            np.linspace(-self._stimSize // 2, self._stimSize // 2 - 1, self._stimSize) +
            0.5,
        )
        self.R, self.P = np.sqrt(self.X ** 2 + self.Y ** 2), np.arctan2(self.Y, self.X)
        self.P[self.P < 0] += 2 * np.pi

        self._stimRaw = np.zeros((self.nFrames, self._stimSize, self._stimSize))

        it = 0
        for cross in self.paradigms:
            for i in range(self.framesPerParadigm):
                frame = np.zeros((self._stimSize, self._stimSize))

                if "w" in cross:
                    pass
                    # minP =
                    # maxP =
                    # frame[np.all((self.P>minP, self.P<maxP),0)] = 1

                elif "r" in cross:
                    pass
                    # minR =
                    # maxR =
                    # frame[np.all((self.R>minR, self.R<maxR),0)] = 1

                self._stimRaw[it, ...] = frame
                it += 1

            it += self.blankLength

        # self.nBars = nBars
        # self.doubleBarRot = doubleBarRot
        # self.thickRatio = thickRatio

        # self.startingDirection = [0,3,6,1,4,7,2,5]
        # self.crossings = len(self.startingDirection)
        # self.framesPerCrossing = 18

        # self.overlap = overlap
        # self.barWidth = np.ceil(self._stimSize / (self.framesPerCrossing * self.overlap - .5)).astype('int')

        # self._stimRaw = np.zeros((self.nFrames, self._stimSize, self._stimSize))
        # self._stimBase = np.zeros(self.nFrames) # to find which checkerboard to use

        # it = 0
        # for cross in self.startingDirection:
        #     for i in range(self.framesPerCrossing):
        #         frame = np.zeros((self._stimSize,self._stimSize))
        #         frame[:, max(0, int(self.overlap*self.barWidth*(i-1))):min(self._stimSize, int(self.overlap*self.barWidth*(i-1)+self.barWidth))] = 1

        #         if self.nBars > 1:
        #             self.nBarShift = self._stimSize // self.nBars
        #             frame2 = np.zeros((self._stimSize,self._stimSize))

        #             for nbar in range(self.nBars-1) :
        #                 o = int(self.overlap*self.barWidth*(i-1)                                     + self.nBarShift*(nbar+1))
        #                 t = int(self.overlap*self.barWidth*(i-1) + self.barWidth*self.thickRatio + self.nBarShift*(nbar+1))
        #                 if o>self._stimSize: o -= self._stimSize
        #                 if t>self._stimSize: t -= self._stimSize
        #                 frame2[:, max(0, o):min(self._stimSize, t)] = 1
        #                 frame2 = skiT.rotate(frame2, self.doubleBarRot, order=0)
        #                 frame = np.any(np.stack((frame,frame2), 2), 2)

        #         self._stimRaw[it,...] = skiT.rotate(frame, cross*360/self.crossings, order=0)

        #         self._stimBase[it] = np.mod(cross,2)+1

        #         it += 1

        #     if cross%2 != 0:
        #         it += self.blankLength

        # self._create_mask(self._stimRaw.shape)

        # self._stimUnc = np.zeros(self._stimRaw.shape)
        # self._stimUnc[:,self._stimMask] = self._stimRaw[:,self._stimMask]

    def _checkerboard(self, nFlickerRings=18, nFlickerWedge=24):
        """create the two flickering main images"""
        self.nFlickerRings = nFlickerRings
        self.nFlickerWedge = nFlickerWedge

        if not hasattr(self, 'R'):
            # define meshgrid in polar corrdinates
            self.X, self.Y = np.meshgrid(
                np.linspace(-self._stimSize // 2, self._stimSize // 2 - 1, self._stimSize) +
                0.5,
                np.linspace(-self._stimSize // 2, self._stimSize // 2 - 1, self._stimSize) +
                0.5,
            )
            self.R, self.P = np.sqrt(self.X ** 2 + self.Y ** 2), np.arctan2(self.Y, self.X)
            self.P[self.P < 0] += 2 * np.pi

        flickerRingsWidth = self._stimSize / (self.nFlickerRings) / 2
        flickerWedgeWidth = 2 * np.pi / self.nFlickerWedge

        self.checkA = np.ones((self._stimSize, self._stimSize))
        self.checkB = np.ones((self._stimSize, self._stimSize))

        for ring in range(self.nFlickerRings):
            for wedge in range(self.nFlickerWedge):
                msk = np.all(
                    (
                        self.R > flickerRingsWidth * ring,
                        self.R < flickerRingsWidth * (ring + 1),
                        self.P > flickerWedgeWidth * wedge,
                        self.P < flickerWedgeWidth * (wedge + 1),
                    ),
                    0,
                )

                self.checkA[msk] = np.mod(ring + wedge, 2)
                self.checkB[msk] = np.mod(ring + wedge + 1, 2)