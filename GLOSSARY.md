# Precision Agriculture & Canopy Health Monitoring

Domain terminology for aerial drone-based multispectral crop monitoring and anomaly detection.

## Language

**Canopy Frame**:
An aerial sensor image capturing crop ground cover, normalized to reflectance values in [0, 1].
_Avoid_: Raw image, battlefield capture, scene

**Multispectral Canopy Frame**:
A calibrated multi-band sensor capture encapsulating raw spectral bands alongside synthesized vegetation indices (NDVI and NDRE).
_Avoid_: Band stack, composite image, multi-layer TIFF

**Canopy Anomaly**:
A localized vegetation region exhibiting significant spectral divergence from healthy crop baseline (e.g. disease, pest damage, irrigation failure, weed invasion).
_Avoid_: Threat, tactical target, generic defect

**Baseline Reconstruction**:
The synthetic healthy canopy image synthesized by the autoencoder representing expected healthy vegetation signature for the observed field.
_Avoid_: Recon, reference frame

**Spectral Difference Heatmap**:
The spatial pixel-wise error matrix quantifying deviation between the restored canopy frame and the baseline reconstruction.
_Avoid_: Error map, battle heatmap

**Anomaly Region**:
A contiguous spatial cluster of anomalous pixels isolated via morphological filtering and Non-Maximum Suppression, specified with spatial coordinates and anomaly score.
_Avoid_: Target box, tactical threat, candidate box

**Flight Telemetry Log**:
An offline-buffered record coupling a detected canopy anomaly with GPS coordinates, flight altitude, timestamp, and calibrated class confidence.
_Avoid_: Threat log, battlefield alert, SQLite entry
