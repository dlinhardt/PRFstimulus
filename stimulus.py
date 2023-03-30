import numpy as np
from skimage.transform import resize
from nipy.modalities.fmri.hrf import spm_hrf_compat
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from numpy.random import randint
import skimage.transform as skiT

import subprocess
from glob import glob
import os
from scipy.io import loadmat, savemat


class Stimulus:
    def __init__(
        self,
        stimSize=51,
        maxEcc=7,
        TR=2,
        stim_duration=336,
        blank_duration=12,
        loadImages=None,
        flickerFrequency=8,
        continous=False,
    ):

        self._maxEcc = maxEcc
        self._stimSize = stimSize
        self.TR = TR
        self._loadImages = loadImages
        self._carrier = "images" if self._loadImages is not None else "checker"
        self._stim_duration = stim_duration

        self.continous = continous

        self.nFrames = np.ceil(stim_duration / self.TR).astype(int)
        self.blankLength = np.ceil(blank_duration / self.TR).astype(int)

        # if np.mod(stim_duration, self.TR) > 0:
        #     print(f'WARNING: stim_duration will be clipped to TR '
        #           f'({self.nFrames*self.TR}s instead of {stim_duration}s)!')
        #     self._stim_duration = self.nFrames * self.TR

        # if np.mod(blank_duration, self.TR) > 0 and not self.continous:
        #     print(f'WARNING: blank_duration will be clipped to TR '
        #           f'({self.blankLength*self.TR}s instead of {blank_duration}s)!')
        #     self._blank_duration = self.blankLength * self.TR


        self.flickerFrequency = flickerFrequency  # Hz

    def save(self):
        pass
        """save class as self.name.txt
            not done yet just dummy"""

        file = open(self.name + ".txt", "w")
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        pass
        """try load self.name.txt
            also not done yet"""

        file = open(self.name + ".txt", "r")
        dataPickle = file.read()
        file.close()

        self.__dict__ = pickle.loads(dataPickle)

    def flickeringStim(self, compress=False):
        """create flickering checkerboard stimulus from binary stimulus mask"""

        if self._stimSize < 512:
            print("You should consider going for higher resolution. e.g. 1024x1024")

        if self._carrier == "checker":
            self._checkerboard()
        elif self._carrier == "images":
            self._loadCarrierImages(self._loadImages)
        else:
            Warning(f"Invalid carrier {self._carrier}, choose from [checker, images]!")

        if not self.continous:
            framesPerPos = int(np.round(self.TR * self.flickerFrequency + .0001))
            nF = self.nFrames

            self._flickerUncStim = np.ones((self._stimSize, self._stimSize, nF * framesPerPos)) * 128
        else:
            framesPerPos = 2
            nF = self.nContinousFrames

            self._flickerUncStim = np.ones((self._stimSize, self._stimSize, nF * 2 + 1)) * 128

        self._flickerSeqTimeing = np.arange(0, self._stimUnc.shape[0] * self.TR, 1 / self.flickerFrequency)

        if self._carrier == "checker":
            for i in range(nF):
                for j in range(framesPerPos):
                    if self._stimBase[i] == 1:
                        checkOne = self.checkA
                        checkTwo = self.checkB
                    elif self._stimBase[i] == 2:
                        checkOne = self.checkC
                        checkTwo = self.checkD

                    if np.mod(j, 2) == 0:
                        self._flickerUncStim[..., framesPerPos * i + j][self._stimUnc[i, ...].astype(bool)
                                                    ] = checkOne[self._stimUnc[i, ...].astype(bool)]
                    elif np.mod(j, 2) == 1:
                        self._flickerUncStim[..., framesPerPos * i + j][self._stimUnc[i, ...].astype(bool)
                                                    ] = checkTwo[self._stimUnc[i, ...].astype(bool)]

        elif self._carrier == "images":
            for i in range(nF):
                for j in range(framesPerPos):
                    self._flickerUncStim[..., framesPerPos * i + j][self._stimUnc[i, ...].astype(bool)
                                                    ] = self.carrierImages[self._stimUnc[i, ...].astype(bool),
                                                           randint(self.carrierImages.shape[-1]),]

        if compress:
            print('Compressing stimulus, this may take some time...')
            # get only the unique images to compress the output mat file
            self._flickerUncStim, self._flickerSeq = np.unique(self._flickerUncStim, axis=-1, return_inverse=True)

            # swap the empty image to the first position
            swap = self._flickerSeq[-1]
            self._flickerUncStim[...,0], self._flickerUncStim[...,swap] = self._flickerUncStim[...,swap], self._flickerUncStim[...,0]
            self._flickerSeq[self._flickerSeq==swap], self._flickerSeq[self._flickerSeq==0] = 0, swap
        else:
            self._flickerSeq = np.zeros(nF*framesPerPos, dtype=int)
            for i in range(nF):
                for j in range(framesPerPos):
                    self._flickerSeq[i * framesPerPos + j] = 2 * i + np.mod(j, 2)

    def saveMrVistaStimulus(self, oName, triggerKey="6", compress=True):
        """save the created stimulus as mrVista _images and _params to present it at the scanner"""

        if not hasattr(self, "_flickerSeq"):
            self.flickeringStim(compress=True)

        self.fixSeq = np.zeros(len(self._flickerSeq))
        chunckSize = 9
        colour = 1 if np.random.rand() > 0.5 else 2
        i = 0
        while i < len(self._flickerSeq):
            self.fixSeq[i : i + chunckSize] = colour
            if self.fixSeq[i - 1] != self.fixSeq[i]:
                self.fixSeq[i : i + 4 * chunckSize] = colour
                i += 4 * chunckSize
            else:
                if np.random.rand() < 0.4:
                    colour = 1 if colour == 2 else 2
                i += chunckSize

        oStim = {
            'images'    : self.flickerUncStim.astype("uint8"),
            'seq'       : (self._flickerSeq + 1).astype("uint16"),
            'seqtiming' : self._flickerSeqTimeing.astype('<f8'),
            'cmap'      : np.vstack((aa := np.linspace(0, 1, 256), aa, aa)).T,
            'fixSeq'    : self.fixSeq.astype("uint8"),
        }

        oPara = {
            'experiment' : 'experiment from file',
            'fixation'   : 'disk',
            'modality'   : 'fMRI',
            'trigger'    : 'scanner triggers computer',
            'period'     : np.float64(len(self._flickerSeq)/self.flickerFrequency),
            'tempFreq'   : np.float64(self.flickerFrequency),
            'tr'         : np.float64(self.TR),
            'scanDuration': np.float64(len(self._flickerSeq)/self.flickerFrequency),
            'saveMatrix' : 'None',
            'interleaves': [],
            'numImages'  : np.float64(self.nFrames),
            'stimSize'   : 'max',
            'stimSizePix': np.float64(self._stimSize),
            'radius'     : np.float64(self._maxEcc),
            'prescanDuration' : np.float64(0),
            'runPriority': np.float64(7),
            'calibration': [],
            'numCycles'  : np.float64(1),
            'repetitions': np.float64(1),
            'motionSteps': np.float64(2),
            'countdown'  : np.float64(0),
            'startScan'  : np.float64(0),
        }

        if hasattr(self, '_onsets'):
            oPara['onsets'] = self._onsets

        if self.stimulus_type == 'bar':
            oPara['barWidthDeg'] = np.float64(np.round(self.barWidth / self._stimSize * self._maxEcc * 2, 2))
            oPara['checkerSize'] = np.float64(np.round(self.checkSize / self._stimSize * self._maxEcc, 2))

        oMat  = {
            'stimulus' : oStim,
            'params'   : oPara,
        }

        # if "/" not in oName:
        # savemat(os.path.join("/home_local/dlinhardt/Dropbox/measurementlaptop/images", oName), oMat)
        # else:
        print(f"Saving {oName}... ")
        savemat(oName, oMat, do_compression=True)
        print(f"saved.")

    def playVid(self, z=None, flicker=False):
        """play the stimulus video, if not defined otherwise, the unconvolved stimulus"""

        if flicker:
            if not hasattr(self, "_flickerSeq"):
                self.flickeringStim()

            plt.figure(constrained_layout=True)
            plt.gca().set_aspect("equal", "box")
            for i in range(len(self._flickerSeq)):
                plt.title(i)
                if i == 0:
                    img_artist = plt.gca().imshow(
                        self.flickerUncStim[..., self._flickerSeq[i]],
                        cmap="Greys",
                        vmin=0,
                        vmax=255,
                    )
                else:
                    img_artist.set_data(self.flickerUncStim[..., self._flickerSeq[i]])
                plt.pause(1 / self.flickerFrequency)
        else:
            if not np.any(z):
                z = self._stimUnc
            plt.figure()

            for i in range(z.shape[0]):
                plt.title(i)
                if i == 0:
                    img_artist = plt.gca().imshow(z[i, ...], cmap="Greys")
                else:
                    img_artist.set_data(z[i, ...])
                plt.pause(0.1)

    def saveVid(self, vPath, vName, z=None):
        """save the stimulus video to given path, if not defined otherwise, the unconvolved stimulus"""
        if not np.any(z):
            z = self._stimUnc
        for i in range(self.nFrames):
            plt.title(i)
            plt.imshow(z[i, ...])
            plt.savefig(vPath + "/file%02d_frame.png" % i, dpi=150)

        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "2",
                "-i",
                vPath + "/file%02d_frame.png",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                vPath + "/" + vName + ".mp4",
            ]
        )
        for file_name in glob(vPath + "/*_frame.png"):
            os.remove(file_name)

    def _cart2pol(self, x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    def _pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def _create_mask(self, shape):
        x0, y0 = shape[1] // 2, shape[1] // 2
        n = shape[1]
        r = shape[1] // 2

        y, x = np.ogrid[-x0 : n - x0, -y0 : n - y0]
        self._stimMask = x * x + y * y <= r * r

    @property
    def xVec(self):
        if not hasattr(self, "x"):
            self.y, self.x = np.meshgrid(
                np.linspace(-self._maxEcc, self._maxEcc, self._stimSize),
                np.linspace(-self._maxEcc, self._maxEcc, self._stimSize),
            )
        return self.x[self._stimMask]

    @property
    def yVec(self):
        if not hasattr(self, "y"):
            self.y, self.x = np.meshgrid(
                np.linspace(-self._maxEcc, self._maxEcc, self._stimSize),
                np.linspace(-self._maxEcc, self._maxEcc, self._stimSize),
            )
        return self.y[self._stimMask]

    @property
    def stimUncOrigVec(self):
        return self._stimUncOrig[:, self._stimMask].T

    @property
    def stimOrigVec(self):
        return self._stimOrig[:, self._stimMask].T

    @property
    def stimUncVec(self):
        if not hasattr(self, "_stim"):
            self.convHRF()
        return self._stimUnc[:, self._stimMask].T

    @property
    def flickerUncStim(self):
        if not hasattr(self, "_flickerUncStim"):
            self.flickeringStim()
        return self._flickerUncStim

    @property
    def stim(self):
        if not hasattr(self, "_stim"):
            self.convHRF()
        return self._stim

    @property
    def stimUnc(self):
        return self._stimUnc

    @property
    def stimVec(self):
        if not hasattr(self, "_stim"):
            self.convHRF()
        return self._stim[:, self._stimMask].T

    def stimulus_length(self):
        print(f'Stimulus length: {self._stim_duration}s')
        print(f'length stim_unc: {self.stimUnc.shape[0]} bar positions')

    # things we need for every analysis

    def convHRF(self):
        """'Convolve stimUnc with SPM HRF"""
        self.hrf = spm_hrf_compat(np.linspace(0, 30, 15 + 1))

        self._stim = np.apply_along_axis(
            lambda m: np.convolve(m, self.hrf, mode="full")[:],
            axis=0,
            arr=self._stimUnc,
        )
        self._stim = self._stim[: self._stimUnc.shape[0], ...]

        if hasattr(self, "_stimUncOrig"):
            self._stimOrig = np.apply_along_axis(
                lambda m: np.convolve(m, self.hrf, mode="full")[:],
                axis=0,
                arr=self._stimUncOrig,
            )
            self._stimOrig = self._stimOrig[: self._stimUncOrig.shape[0], ...]

    # different artificial scotomas

    def centralScotoma(self, scotSize):
        """Mask stimulus with central round scotoma"""

        self._stimUncOrig = deepcopy(self._stimUnc)

        mask = np.ones((self._stimSize, self._stimSize))

        x0, y0 = self._stimSize // 2, self._stimSize // 2
        n = self._stimSize
        r = self._stimSize // 2 * scotSize // self._maxEcc

        y, x = np.ogrid[-x0 : n - x0, -y0 : n - y0]
        mask = x * x + y * y >= r * r

        self._stimUnc *= mask[None, ...]

    def peripheralScotoma(self, scotSize):
        """Mask stimulus with peripheral part missing"""

        self._stimUncOrig = deepcopy(self._stimUnc)

        mask = np.ones((self._stimSize, self._stimSize))

        x0, y0 = self._stimSize // 2, self._stimSize // 2
        n = self._stimSize
        r = self._stimSize // 2 * scotSize // self._maxEcc

        y, x = np.ogrid[-x0 : n - x0, -y0 : n - y0]
        mask = x * x + y * y <= r * r

        self._stimUnc *= mask[None, ...]

    def quaterout(self):
        """Mask stimulus with one quater missing all the time"""

        self._stimUncOrig = deepcopy(self._stimUnc)

        mask = np.ones((self._stimSize, self._stimSize))

        for i in range(self._stimSize):
            for j in range(self._stimSize):
                mask[i, j] = (
                    0
                    if np.all((i > self._stimSize // 2, j > self._stimSize // 2))
                    else 1
                )

        self._stimUnc *= mask[None, ...]

    def verification(self):
        """Mask stimulus so that Quadrants:
        IV:  2 stimulations
        III: 4 stimulations
        II:  6 stimulations
        I:   8 stimulations
        """
        self._stimUncOrig = deepcopy(self._stimUnc)

        maskQ1 = np.ones((self._stimSize, self._stimSize))
        maskQ2 = np.ones((self._stimSize, self._stimSize))
        maskQ3 = np.ones((self._stimSize, self._stimSize))

        for i in range(self._stimSize):
            for j in range(self._stimSize):
                maskQ1[i, j] = (
                    0
                    if np.all((i > self._stimSize // 2, j > self._stimSize // 2))
                    else 1
                )
                maskQ2[i, j] = 0 if i > self._stimSize // 2 else 1
                maskQ3[i, j] = (
                    0
                    if np.any((j < self._stimSize // 2, i > self._stimSize // 2))
                    else 1
                )

        # 0-18, 18-36, 42-60, 60-78, 84-102, 102-120, 126-144, 144-162

        for i in range(int(self.nFrames)):
            if i >= 0 and i < self.framesPerCrossing:
                pass
            elif i >= self.framesPerCrossing and i < self.framesPerCrossing * 2:
                self._stimUnc[i, ...] *= maskQ1
            elif (
                i >= self.framesPerCrossing * 2 + self.blankLength and
                i < self.framesPerCrossing * 3 + self.blankLength
            ):
                self._stimUnc[i, ...] *= maskQ2
            elif (
                i >= self.framesPerCrossing * 3 + self.blankLength and
                i < self.framesPerCrossing * 4 + self.blankLength
            ):
                self._stimUnc[i, ...] *= maskQ3
            elif (
                i >= self.framesPerCrossing * 4 + self.blankLength * 2 and
                i < self.framesPerCrossing * 5 + self.blankLength * 2
            ):
                pass
            elif (
                i >= self.framesPerCrossing * 5 + self.blankLength * 2 and
                i < self.framesPerCrossing * 6 + self.blankLength * 2
            ):
                self._stimUnc[i, ...] *= maskQ1
            elif (
                i >= self.framesPerCrossing * 6 + self.blankLength * 3 and
                i < self.framesPerCrossing * 7 + self.blankLength * 3
            ):
                self._stimUnc[i, ...] *= maskQ2
            elif (
                i >= self.framesPerCrossing * 7 + self.blankLength * 3 and
                i < self.framesPerCrossing * 8 + self.blankLength * 3
            ):
                self._stimUnc[i, ...] *= maskQ3

    def _loadCarrierImages(self, loadImages):
        if loadImages.endswith(".mat"):
            self.carrierImages = loadmat(loadImages, simplify_cells=True)["images"][
                :, :, 1, :
            ]

            # resize them if necessary
            if (
                self.carrierImages.shape[0] != self._stimSize or
                self.carrierImages.shape[1] != self._stimSize
            ):
                self.carrierImages = resize(
                    self.carrierImages,
                    (self._stimSize, self._stimSize),
                    anti_aliasing=True,
                )

            # rescale them to [0,255]
            if self.carrierImages.min() != 0:
                self.carrierImages += self.carrierImages.min()

            if self.carrierImages.max() != 255:
                if self.carrierImages.max() != 1:
                    self.carrierImages %= self.carrierImages.max()
                self.carrierImages *= 255
                self.carrierImages = self.carrierImages.astype(int)

        else:
            Warning("Please provide carrier images as .mat file!")
