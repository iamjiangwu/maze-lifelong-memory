# Extended Data Figure 8. Reward code stability across sampling and route controls

Panels a–c contain saved spatial-decoder and PV-location controls. Panels d–j show mouse-level direction, reliability, equal-frame, reward-delivery, alternate-start and cross-direction controls. Physical reward node 94 and decoder bin 103 are distinct.

Balanced decoding averages pairs within animal and uses four exact one-sided sign-flip tests with BH4. PV recurrence uses animal-clustered BH2. Route intervals are Student-t intervals across mice. The eight tests in e, f and i form BH8; g and j each use one separate exact two-sided mouse sign-flip test.

Run from the repository root:

```powershell
python figures/extended_data_08/build.py --output-dir outputs/ed8
```

The release starts at approved saved derived observations and display arrays. Neural values use the accepted S/HWHM 1 configuration and 0.066 eligibility gate where applicable. Metric-specific eligibility exceptions remain in the input tables. Missing values and support masks are preserved.

The route controls use unsmoothed S (HWHM 0), with within-neuron route standardization applied upstream. Their saved mouse-level observations are used directly here.

[Data](data/README.md) · [Code](code/README.md) · [Requirements](../../docs/REPRODUCING.md)
