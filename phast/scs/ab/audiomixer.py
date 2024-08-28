import numpy as np

# [wavOut clip] = audioMixerUnit(wav_1, ..., wav_n, par)
# Mix arbitrary number of audio inputs signals wav_i. For every signal, the
# target level as well as the target level type have to be specified
# (abs. rms/abs. peak/rel. to input). par.sensIn defines the peak SPL
# equivalent to 0dB FS. par.wrap controls the behaviour of input signals
# are of unequal length (warp around or zero-pad to match duration of the
# primary signal). For par.wrap = 1, par.durFade controls the duration of
# a cosine-shaped cross-fade between the end and beginning.
#
# INPUT:
#   wav_i - vector containing wav data for input channel i
#   par   - paramteter struct / object
#
# FIELDS FOR PAR:
#   parent.fs    - input sampling rate
#   sensIn   - input sensitivity: dB SPL corresponding to digital RMS amp. (0dB re.) 1
#                  [equivalently: dB peak corresponding to digital peak amp. 1]
#   lvlType  - string or cell array of strings: 'rms' / 'peak' / 'rel';
#              if string, same type is used for all channels;
#              if cell aray, the size must match n
#   lvlDb    - scalar or n-vector of levels per channel in dB; for types 'rms'
#              and 'peak', lvlDb(i) is in dB SPL; for type 'rel', lvlDb(i) is
#              in dB relative to the input level.
#              If a scalar is provided, the gain derived for the primary
#              input (see below) is applied to all channels equally. %   delays   - vector of onset delays for each input [s]
#   primaryIn - which input determines the length of the mixed output
#               (1..nInputs, or []); if [], the longest input (including
#               delay) is chosen as primary.
#   wrap     - repeat shorter inputs to match duration of the primary
#              input? [1/0]
#   durFade  -  duration of cross-fade when wrapping signals [s]
#   channelMode - treat each channel independently, or apply gain
#                 determined for the primary input to all channels?
#                 ['independent' / 'primary']
#
# OUTPUT:
#    wavOut  - mixed signal (column vector)
#    clip    - clipping indicator [0/1]
#
# Copyright (c) 2012-2020 Advanced Bionics. All rights reserved.


def audiomixer(
    wav_file,
    lvlType="rms",
    Fs=17400,
    delays=[],
    primaryIn=[],
    sensIn=111.6,
    lvlDb=65,
    clipValue=1,
    durFade=0,
    **kwargs,
):  #
    # The results differ slightly from the matlab file because the wav file is already different somehow

    # par = varargin{end};

    # NOT NECESSARY ONLY WANT TO CHANGE DB
    # nWav = nargin-1;
    # wav = varargin(1:nWav);
    # nChannels = zeros(nWav, 1);
    # # make all wavs column vectors and determine number of channels
    # for iWav = 1:nWav:
    #     [M, N] = size(wav{iWav});
    #     if M>N
    #         nChannels(iWav) = N;
    #         wav{iWav} = reshape(wav{iWav}', M*N, 1);
    #     else
    #         nChannels(iWav) = M;
    #         wav{iWav} = reshape(wav{iWav}, M*N, 1);
    nChannels = 1
    nWav = 1

    # # ensure all inputs have the same number of channels
    # assert(all(nChannels==nChannels(1)), 'All inputs need to have the same number of channels.')

    # primaryIn = par.primaryIn;

    # assert(all(cellfun(@(X__) isnumeric(X__) & isvector(X__), wav(1:nWav))), 'wav_1..wav_n must be numerical vectors')
    # assert(length(lvlDb) == nWav || isscalar(lvlDb), 'Length of lvlDb must equal the number of audio inputs.');
    # assert(isempty(delays) || (length(delays) == nWav), 'Length of par.delays must 0 or equal the the number of audio inputs. ' )
    # assert(ischar(lvlType) || iscellstr(lvlType), 'lvlType must be a string of cell array of strings.')
    # assert(isempty(primaryIn) || (~mod(primaryIn,1) && (primaryIn <= nWav) && primaryIn > 0),...
    #     'primaryIn must be empty or an integer less or equal to the number of audio inputs.');

    # compute onset delay for each audio input
    if delays:
        assert delays >= 0, "Elements of delays must be non-negative."
        delays = nChannels * round(delays * Fs)
    else:
        delays = 0

    # # get level type for each input
    # if ischar(par.lvlType)
    #     lvlType = repmat({par.lvlType},1,nWav);
    # else
    #     lvlType = par.lvlType;
    # end

    # determine input signal lengths
    if len(wav_file) == 1:
        lenWavIn = len(wav_file[0])
    else:
        lenWavIn = len(wav_file)
    lenWavDelayed = lenWavIn + delays  # input length including delays

    # determine output length
    # if primaryIn:
    #     lenOut = max(lenWavDelayed)
    # else:
    #     lenOut = lenWavDelayed(primaryIn)
    lenOut = lenWavDelayed

    # length of cross-fade in samples, and fade-in/out envelopes
    lenFade = np.ceil(durFade * Fs)
    envFadeOut = 0.5 * np.transpose(np.cos(np.linspace(0, np.pi, int(lenFade)))) + 0.5
    # Adjust fade to account for number of channels and their arrangement in the colum vector
    # envFadeOut = reshape(repmat(envFadeOut,1,nChannels)',1,lenFade*nChannels)'
    lenFade = lenFade * nChannels
    envFadeIn = 1 - envFadeOut

    # determine input levels (prior to padding/wrapping)
    # lvlWav = NaN(1,nWav)
    # for iWav = 1:nWav:
    # switch lower(lvlType{iWav})
    if lvlType.lower() == "rms":
        lvlWav = 10 * np.log10(np.mean(wav_file[0] ** 2)) + sensIn
    elif lvlType.lower() == "peak":
        lvlWav = 20 * np.log10(max(abs(wav_file[0]))) + sensIn
    elif lvlType.lower() == "rel":
        lvlWav = 0
    #             otherwise
    #                 error('Unknown level scaling type ''%s'' at index %d', lvlType{iWav}, iWav);

    #     # find wavs that need to be wrapped / padded
    #     needsLengthening = (lenWavDelayed < lenOut)

    #     for iWav = 1:nWav:
    #         # RETHINK nRep with delays!
    #         if needsLengthening(iWav)
    #             if par.wrap % wrap signal
    #                 nRep = ceil( (lenOut-delays(iWav))/(lenWavIn(iWav)-lenFade) - 1 );
    #                 wavCross = envFadeOut .* wav{iWav}(end-lenFade+1:end) + envFadeIn .* wav{iWav}(1:lenFade);
    #                 wav{iWav} = [zeros(delays(iWav),1); wav{iWav}(1:end-lenFade); ...
    #                     repmat([wavCross; wav{iWav}(lenFade+1:end-lenFade)], nRep, 1 )];
    #                 wav{iWav}(lenOut+1:end) = [];
    #             else % zero-pad signal
    #                 wav{iWav} = [zeros(delays(iWav),1); wav{iWav}; zeros(lenOut-lenWavDelayed(iWav),1)];
    #             end
    #         else % truncate signal
    #             wav{iWav} = [zeros(delays(iWav),1); wav{iWav}(1:lenOut-delays(iWav))];
    #             wav{iWav}(end+1:lenOut) = [];

    #     # keyboard
    #     assert(all(cellfun(@(W__) length(W__) == lenOut, wav)), 'All wavs must have length lenMax by now.')

    # compute gain in dB for each input
    if (
        np.isscalar(lvlDb) and primaryIn == []
    ):  # master gain applied to all inputs, but not primary input specified
        gains = (lvlDb - lvlWav) * np.ones(nWav)
    elif np.isscalar(lvlDb) and np.isscalar(
        primaryIn
    ):  # master gain applied to all inputs, derived from primary input
        gains = (lvlDb[primaryIn] - lvlWav[primaryIn]) * np.ones(nWav)
    # else: #gain computed independently for each channel
    # gains = np.zeros(nWav)
    # for iWav = 1:nWav:
    # gains(iWav) = (par.lvlDb(iWav)-lvlWav(iWav));

    # add scaled inputs
    wavOut = np.zeros((lenOut, 1))
    # for iWav = 1:nWav
    wavOut = wavOut + np.transpose(wav_file) * 10 ** (gains / 20)
    wav = wav_file * 10 ** (gains / 20)

    # check clipping
    maxAbsOut = max(abs(wavOut))
    clip = maxAbsOut > clipValue
    if clip:
        print(
            "Clipping occured. Maximum output amplitude %.2f (%.2fdB FS, equiv. %.2fdB SPL)",
            maxAbsOut,
            20 * np.log10(maxAbsOut),
            20 * np.log10(maxAbsOut) + sensIn,
        )

    # undo flattening
    # wavOut = reshape(wavOut, nChannels, lenOut/nChannels).'
    #     # same on all input signals
    #     for iWav = 1:nWav:
    #         wav{iWav} = reshape(wav{iWav}, nChannels, lenOut/nChannels).'
    wavOut = np.transpose(wavOut)  # should be of shape (1xlen(signal))

    return wavOut, wav, clip


# from frontend import read_wav
# wav_file, *_ = read_wav('C:/Users/Savin/PycharmProjects/temporal/abt/sounds/AzBio_3sent.wav')
# wavOut, wav, clip = audiomixer(wav_file)
