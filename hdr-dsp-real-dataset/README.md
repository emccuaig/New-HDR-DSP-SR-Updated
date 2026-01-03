
# Dataset description

It contains 2660 L1A sequences (folder `crop/`) of size 128x128 (uint16).
The length of each sequence varies from 6 to 15 frames.
Each sequence is produced by registering the frames with integer translations. This means that the frames are purposefully not exactly aligned since it would require resampling of the data.

Associated to each sequence, there are the saturation masks (`satmask/`) and exposure ratios (`ratios/`).
The saturation mask for a frame is a boolean mask, equals to 1 where the sample is valid, 0 otherwise (due to saturation).
The exposure ratios correspond to the exposure times per frame in millisecond as given by Planet as metadata. By dividing a sequence by the ratios vector (one float per frame), the radiometry of the resulting sequence should be approximately equalized. However, due to imprecision in these exposure times, this equalization is not perfect.

Refer to the paper and its supplementary material for more details.

