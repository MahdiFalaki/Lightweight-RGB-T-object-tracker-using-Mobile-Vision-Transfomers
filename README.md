# Lightweight RGB-T Tracking with Mobile Vision Transformers

**Mahdi Falaki, Maria A. Amer**  
Department of Electrical and Computer Engineering, Concordia University, Montréal, Québec, Canada  

**Submitted to ICASSP 2026.**  
**Code will be released after the review period.**

---

## Highlights

![The pipeline of proposed RGB-T tracker](Pipeline.png)  
*Figure 1: Pipeline of the proposed RGB-T tracker.*

![Architecture of mmMobileViT](mmMobileViT.png)  
*Figure 1: Pipeline of the proposed RGB-T tracker.*

![Additional visual material from the submission](ablation.png)  
*Figure 2: Visual material from the submission.*

*Table 1: Comparison on LasHeR, RGBT234, and GTOT.*

| Tracker | #Params (M) | MACs (G) | FPS (GPU) | LasHeR PR | LasHeR SR | LasHeR NPR | RGBT234 MPR | RGBT234 MSR | GTOT PR | GTOT SR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SUTrack Tiny | 22 | 3 | 100 | 0.667 | 0.539 | – | 0.859 | 0.638 | 0.853 | 0.726 |
| EMTrack | 16 | 2 | 83.8 | 0.659 | 0.533 | – | 0.838 | 0.601 | – | – |
| CMD | 19.9 | – | 30 | 0.590 | 0.464 | 0.546 | 0.824 | 0.584 | 0.892 | 0.734 |
| TBSI Tiny | 14.9 | – | 40 | 0.617 | 0.489 | 0.578 | 0.794 | 0.555 | 0.881 | 0.706 |
| **Ours** | **3.93** | **4.35** | **121.9** | 0.603 | 0.473 | 0.567 | 0.806 | 0.589 | 0.895 | 0.7467 |
| SMAT* (RGB-only) | 3.76 | – | 154.6 | 0.549 | 0.438 | 0.512 | 0.737 | 0.536 | 0.690 | 0.578 |

---

## Attribute-based analysis (RGBT234)

![RGBT234 MPR radar](RGBT234_MPR_radar.png)  
*MPR attribute-wise analysis on RGBT234.*

![RGBT234 MSR radar](RGBT234_MSR_radar.png)  
*MSR attribute-wise analysis on RGBT234.*

**Attributes (RGBT234):**  
NO (No Occlusion), PO (Partial Occlusion), HO (Hyaline Occlusion), LI (Low Illumination), LR (Low Resolution), TC (Thermal Crossover), DEF (Deformation), FM (Fast Motion), SV (Scale Variation), MB (Motion Blur), CM (Camera Moving), BC (Background Clutter).

---

## Citation

```bibtex
@inproceedings{falaki2026lightweight_rgbt_mobilevit,
  title     = {Lightweight RGB-T Tracking with Mobile Vision Transformers},
  author    = {Falaki, Mahdi and Amer, Maria A.},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026},
  note      = {Submitted}
}
