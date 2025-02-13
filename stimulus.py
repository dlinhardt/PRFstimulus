import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle
from numpy.random import randint
import skimage.transform as skiT
from copy import deepcopy
import sys

import subprocess
from glob import glob
import os
from scipy.io import loadmat, savemat

try:
    from nipy.modalities.fmri.hrf import spm_hrf_compat
except:
    print("could not load nipy HRF!")
    print("we will use the function copied from github!")
    """
    Copyright (c) 2006-2021, NIPY Developers
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

        * Neither the name of the NIPY Developers nor the names of any
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."""
    import scipy.stats as sps

    def spm_hrf_compat(
        t,
        peak_delay=6,
        under_delay=16,
        peak_disp=1,
        under_disp=1,
        p_u_ratio=6,
        normalize=True,
    ):
        """SPM HRF function from sum of two gamma PDFs

        This function is designed to be partially compatible with SPMs `spm_hrf.m`
        function.

        The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
        `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
        and dispersion `under_disp`, and divided by the `p_u_ratio`).

        Parameters
        ----------
        t : array-like
            vector of times at which to sample HRF.
        peak_delay : float, optional
            delay of peak.
        under_delay : float, optional
            delay of undershoot.
        peak_disp : float, optional
            width (dispersion) of peak.
        under_disp : float, optional
            width (dispersion) of undershoot.
        p_u_ratio : float, optional
            peak to undershoot ratio.  Undershoot divided by this value before
            subtracting from peak.
        normalize : {True, False}, optional
            If True, divide HRF values by their sum before returning.  SPM does this
            by default.

        Returns
        -------
        hrf : array
            vector length ``len(t)`` of samples from HRF at times `t`.

        Notes
        -----
        See ``spm_hrf.m`` in the SPM distribution.
        """
        if len([v for v in [peak_delay, peak_disp, under_delay, under_disp] if v <= 0]):
            raise ValueError("delays and dispersions must be > 0")
        # gamma.pdf only defined for t > 0
        hrf = np.zeros(t.shape, dtype=np.float64)
        pos_t = t[t > 0]
        peak = sps.gamma.pdf(pos_t, peak_delay / peak_disp, loc=0, scale=peak_disp)
        undershoot = sps.gamma.pdf(
            pos_t, under_delay / under_disp, loc=0, scale=under_disp
        )
        hrf[t > 0] = peak - undershoot / p_u_ratio
        if not normalize:
            return hrf
        return hrf / np.sum(hrf)


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
        """Create flickering stimulus with cleaner structure."""
        if self._stimSize < 512:
            print("Consider using a higher resolution, e.g. 1024x1024")

        # Load carrier images or generate checkerboard pattern.
        if self._carrier == "checker":
            self._checkerboard()
        elif self._carrier == "images":
            self._loadCarrierImages(self._loadImages)
        else:
            Warning(f"Invalid carrier {self._carrier}, choose from [checker, images]!")

        framesPerPos, nF = self._calculate_frames()

        self._init_flickerStim(nF, framesPerPos)

        if self._carrier == "checker":
            self._apply_checker_flicker(nF, framesPerPos)
        elif self._carrier == "images":
            self._apply_image_flicker(nF, framesPerPos)

        if compress:
            self._compress_stimulus()
        else:
            self._set_flicker_sequence(nF, framesPerPos)

    def _calculate_frames(self):
        """Calculate frames per position and total frame count."""
        if not self.continous:
            framesPerPos = int(np.round(self.TR * self.flickerFrequency + 1e-4))
            nF = self.nFrames
        else:
            framesPerPos = int(
                np.round(self.TR / self.frameMultiplier * self.flickerFrequency + 1e-4)
            )
            nF = self.nContinousFrames
        if framesPerPos == 0:
            print("Invalid configuration: framesPerPos computed as 0.")
            print(
                f"continous_multiplier ({self.frameMultiplier / self.TR}) is not allowed to be greater than flickerFrequency ({self.flickerFrequency})"
            )
        return framesPerPos, nF

    def _init_flickerStim(self, nF, framesPerPos):
        """Initialize flickering stimulus and sequence timing arrays."""
        self._flickerUncStim = (
            np.ones((self._stimSize, self._stimSize, nF * framesPerPos)) * 128
        )
        self._flickerSeqTimeing = np.arange(
            0, self._stimUnc.shape[0] * self.TR, 1 / self.flickerFrequency
        )

    def _get_checker_values(self, stim_val):
        if stim_val == 1:
            return self.checkA, self.checkB
        elif stim_val == 2:
            return self.checkC, self.checkD
        else:
            return None, None

    def _apply_checker_flicker(self, nF, framesPerPos):
        """Apply flicker pattern for checkerboard stimuli."""
        running_counter = 0
        for i in range(nF):
            mask = self._stimUnc[i, ...].astype(bool)
            checkOne, checkTwo = self._get_checker_values(self._stimBase[i])

            for j in range(framesPerPos):
                idx = framesPerPos * i + j
                if running_counter % 2 == 0:
                    if checkOne is not None:
                        self._flickerUncStim[..., idx][mask] = checkOne[mask]
                else:
                    if checkTwo is not None:
                        self._flickerUncStim[..., idx][mask] = checkTwo[mask]
                running_counter += 1

    def _apply_image_flicker(self, nF, framesPerPos):
        """Apply flicker pattern for image-based stimuli."""
        for i in range(nF):
            mask = self._stimUnc[i, ...].astype(bool)
            for j in range(framesPerPos):
                idx = framesPerPos * i + j
                rand_idx = randint(self.carrierImages.shape[-1])
                self._flickerUncStim[..., idx][mask] = self.carrierImages[
                    mask, rand_idx
                ]

    def _compress_stimulus(self):
        """Compress the stimulus output."""
        print("Compressing stimulus, this may take some time...")
        self._flickerUncStim, self._flickerSeq = np.unique(
            self._flickerUncStim, axis=-1, return_inverse=True
        )
        # Move the empty image to the first position.
        swap = self._flickerSeq[-1]
        a = deepcopy(self._flickerUncStim[..., swap])
        b = deepcopy(self._flickerUncStim[..., 0])
        self._flickerUncStim[..., 0], self._flickerUncStim[..., swap] = a, b
        self._flickerSeq[self._flickerSeq == swap] = -1
        self._flickerSeq[self._flickerSeq == 0] = swap
        self._flickerSeq[self._flickerSeq == -1] = 0

    def _set_flicker_sequence(self, nF, framesPerPos):
        """Set the flicker sequence if compression is not applied."""
        self._flickerSeq = np.zeros(nF * framesPerPos, dtype=int)
        for i in range(nF):
            for j in range(framesPerPos):
                self._flickerSeq[i * framesPerPos + j] = 2 * i + (j % 2)

    def _create_fixation_sequence(self, seq_length, chunkSize=9):
        """Create the fixation sequence based on the flicker sequence length."""
        fixSeq = np.zeros(seq_length)
        colour = 1 if np.random.rand() > 0.5 else 2
        i = 0
        while i < seq_length:
            fixSeq[i : i + chunkSize] = colour
            if i > 0 and fixSeq[i - 1] != fixSeq[i]:
                fixSeq[i : i + 4 * chunkSize] = colour
                i += 4 * chunkSize
            else:
                if np.random.rand() < 0.4:
                    colour = 1 if colour == 2 else 2
                i += chunkSize
        return fixSeq.astype("uint8")

    def _prepare_output(self):
        """Prepare output dictionaries for saving stimulus."""
        oStim = {
            "images": np.uint8(self.flickerUncStim),
            "seq": np.uint16(self._flickerSeq + 1),
            "seqtiming": np.float64(self._flickerSeqTimeing),
            "cmap": np.vstack((np.linspace(0, 1, 256),) * 3).T,
            "fixSeq": self.fixSeq,
        }
        oPara = {
            "experiment": "experiment from file",
            "fixation": "disk",
            "modality": "fMRI",
            "trigger": "scanner triggers computer",
            "period": np.float64(len(self._flickerSeq) / self.flickerFrequency),
            "tempFreq": np.float64(self.flickerFrequency),
            "tr": np.float64(self.TR),
            "scanDuration": np.float64(len(self._flickerSeq) / self.flickerFrequency),
            "saveMatrix": "None",
            "interleaves": [],
            "numImages": np.float64(self.nFrames),
            "stimSize": "max",
            "stimSizePix": np.float64(self._stimSize),
            "radius": np.float64(self._maxEcc),
            "prescanDuration": np.float64(0),
            "runPriority": np.float64(7),
            "calibration": [],
            "numCycles": np.float64(1),
            "repetitions": np.float64(1),
            "motionSteps": np.float64(2),
            "countdown": np.float64(0),
            "startScan": np.float64(0),
        }
        if hasattr(self, "_onsets"):
            oPara["onsets"] = self._onsets
        if self.stimulus_type == "bar":
            oPara["barWidthDeg"] = np.float64(
                np.round(self.bar_width / self._stimSize * self._maxEcc * 2, 2)
            )
            if self._loadImages is None:
                oPara["checkerSize"] = np.float64(
                    np.round(self.checkSize / self._stimSize * self._maxEcc, 2)
                )
        return oStim, oPara

    def saveMrVistaStimulus(self, oName, triggerKey="6", compress=True):
        """
        Save the created stimulus as mrVista _images and _params to present it at the scanner.
        """
        if not hasattr(self, "_flickerSeq"):
            self.flickeringStim(compress=True)

        self.fixSeq = self._create_fixation_sequence(len(self._flickerSeq))
        oStim, oPara = self._prepare_output()
        oMat = {
            "stimulus": oStim,
            "params": oPara,
        }

        print(f"Saving {oName}... ")
        savemat(oName, oMat, do_compression=True)
        print("saved.")

    def playVid(self, z=None, flicker=False):
        """play the stimulus video, if not defined otherwise, the unconvolved stimulus"""

        if flicker:
            if not hasattr(self, "_flickerUncSeq"):
                self.flickeringStim(compress=True)

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

    def saveVid(self, vPath, vName, flicker=False, z=None):
        """save the stimulus video to given path, if not defined otherwise, the unconvolved stimulus"""
        if not np.any(z):
            if flicker:
                z = np.transpose(self.flickerUncStim, (2, 0, 1))
            else:
                z = self._stimUnc
        for i in range(z.shape[0]):
            plt.title(i)
            plt.imshow(z[i, ...], cmap="Greys")
            plt.savefig(vPath + "/file%02d_frame.png" % i, dpi=150)

        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "12",
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
        rho = np.sqrt(x**2 + y**2)
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
        return self._stimMask

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
        return self.stim[:, self._stimMask].T

    def stimulus_length(self):
        print(f"Stimulus length: {self._stim_duration}s")
        print(f"length stim_unc: {self.stimUnc.shape[0]} bar positions")

    # things we need for every analysis

    def convHRF(self):
        """'Convolve stimUnc with SPM HRF"""
        # if 'nipy.modalities.fmri.hrf' in sys.modules:
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

    def verification(self, version=1):
        """Mask stimulus so that Quadrants:
        IV:  2 stimulations
        III: 4 stimulations
        II:  6 stimulations
        I:   8 stimulations

        sweeps: horizontal 0, 4
                tl - br    1, 5
                vertical   2, 6
                tr - bl    3, 7
        """
        if not self.stimulus_type:
            Warning("Only use this with bar stimuli!")

        self._stimUncOrig = deepcopy(self._stimUnc)

        size = self._stimSize
        sizeHalf = size // 2

        # create masks for single quadrants
        Q1 = np.ones((size, size))
        Q2 = np.ones((size, size))
        Q3 = np.ones((size, size))
        Q4 = np.ones((size, size))

        Q1[sizeHalf:, sizeHalf:] = 0
        Q2[sizeHalf:, :sizeHalf] = 0
        Q3[:sizeHalf, :sizeHalf] = 0
        Q4[:sizeHalf, sizeHalf:] = 0

        cross = np.zeros((self.crossings, self.nFrames))
        for j in range(self.crossings):
            cross[
                j,
                int(self.framesPerCrossing * (j) + self.blankLength * int(j / 2)) : int(
                    self.framesPerCrossing * (j + 1)
                    + self.blankLength * int(j / 2)
                    + 0.5
                ),
            ] = 1

        cross = cross.astype(bool)

        if version == 1:
            # for sweep 1 and 5 mask nothing

            # for sweep 2 and 6 mask Q1
            self._stimUnc[cross[1], ...] *= Q1
            self._stimUnc[cross[5], ...] *= Q1

            # for sweep 3 and 7 mask Q1 and Q2
            self._stimUnc[cross[2], ...] *= np.all((Q1, Q2), 0)
            self._stimUnc[cross[6], ...] *= np.all((Q1, Q2), 0)

            # for sweep 4 and 8 mask Q1, Q2 and Q3
            self._stimUnc[cross[3], ...] *= np.all((Q1, Q2, Q3), 0)
            self._stimUnc[cross[7], ...] *= np.all((Q1, Q2, Q3), 0)

        elif version == 2:
            # for sweep 1 and 5 mask nothing
            self._stimUnc[cross[0], ...] *= Q2
            self._stimUnc[cross[4], ...] *= Q2

            # for sweep 2 and 6 mask Q1
            self._stimUnc[cross[1], ...] *= np.all((Q1, Q2, Q3), 0)
            self._stimUnc[cross[5], ...] *= np.all((Q1, Q2, Q3), 0)

            # for sweep 3 and 7 mask Q1 and Q2
            self._stimUnc[cross[2], ...] *= Q1
            self._stimUnc[cross[6], ...] *= Q1

            # for sweep 4 and 8 mask Q1, Q2 and Q3
            self._stimUnc[cross[3], ...] *= np.all((Q1, Q2, Q3), 0)
            self._stimUnc[cross[7], ...] *= np.all((Q1, Q2, Q3), 0)

    def _loadCarrierImages(self, loadImages):
        if loadImages.endswith(".mat"):
            self.carrierImages = loadmat(loadImages, simplify_cells=True)["images"][
                :, :, 1, :
            ]

            # resize them if necessary
            if (
                self.carrierImages.shape[0] != self._stimSize
                or self.carrierImages.shape[1] != self._stimSize
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
