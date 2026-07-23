# QAC-UNet

## **Quantum Adaptive Coefficient U-Net for Breast Ultrasound Segmentation**

> This paper is currently under review at **ACIVS 2026**, an **ICORE B-ranked conference**.

## **Introduction**

Breast ultrasound is widely used for breast lesion assessment because it is non-invasive, accessible, and cost-effective. However, automatic lesion segmentation remains difficult due to:

- Speckle noise and imaging artifacts
- Low contrast between lesions and surrounding tissue
- Weak or ambiguous lesion boundaries
- Large variations in lesion size, shape, and appearance
- Limited annotated medical datasets

Most segmentation networks apply the same feature-processing behavior to every input image. This fixed behavior may be insufficient when ultrasound images have substantially different intensity distributions, lesion characteristics, and noise patterns.

To address this limitation, this paper proposes **QAC-UNet**, a hybrid CNN–Transformer segmentation framework that uses compact **Variational Quantum Circuits (VQCs)** to generate adaptive coefficients for each input image.

## **Proposed Method**

QAC-UNet extends a HAU-Net-style hybrid CNN–Transformer architecture and introduces three image-conditioned quantum modules:

1. **Quantum-Generated Instance Normalization (QGIN)**  
   Generates image-specific scaling and shifting coefficients for bottleneck normalization.

2. **Quantum Skip Coefficient Modulation (QSCM)**  
   Recalibrates encoder skip features using both skip-level details and bottleneck semantic context.

3. **Quantum Spline-Adaptive Gate (QSAG)**  
   Combines a Kolmogorov–Arnold Network (KAN) with a VQC to control the overall strength of the bottleneck representation.

The full feature maps remain in the classical network. Only compact, low-dimensional descriptors are sent to the quantum circuits, reducing quantum-resource requirements and making the framework more compatible with the **NISQ era**.

<p align="center">
  <img src="Fig1_QAC_UNet_Overall_Architecture.png" width="900" alt="Overall architecture of QAC-UNet"/>
</p>

## **Overall Architecture**

For an input ultrasound image, a ResNet34-style encoder extracts hierarchical features at five resolutions. Local–Global Transformer blocks refine intermediate encoder features, while an Atrous Spatial Pyramid Pooling module captures multi-scale context from the deepest representation.

The processing pipeline can be summarized as follows:

- The encoder extracts local and hierarchical image features.
- Local–Global Transformer blocks model broader contextual relationships.
- ASPP constructs a multi-scale bottleneck representation.
- QGIN performs image-specific bottleneck normalization.
- QSAG controls the global strength of the bottleneck.
- QSCM recalibrates skip features before decoder fusion.
- Cross-Attention Blocks inject deeper semantic information into high-resolution decoder stages.
- The decoder reconstructs the final lesion mask.

## **Main Contributions**

### **1. Image-Conditioned Quantum Coefficients**

QAC-UNet does not directly process large image tensors with a quantum circuit. Instead, compact VQCs receive low-dimensional descriptors and generate adaptive coefficients that regulate selected operations in the classical segmentation network.

This strategy provides sample-specific adaptation while maintaining practical computational requirements.

### **2. Quantum-Generated Instance Normalization**

Ultrasound images from different patients, devices, and acquisition conditions may produce bottleneck features with different statistical distributions.

For a bottleneck feature map \(X\), QGIN computes the spatial mean and variance of each channel. The statistics are projected into a compact descriptor and processed by a VQC to generate channel-wise coefficients \(\gamma_c\) and \(\beta_c\).

```math
\widetilde{X}_{c,h,w}
=
\frac{X_{c,h,w}-\mu_c}
{\sqrt{v_c+\epsilon}}
\left(1+\gamma_c\right)
+
\beta_c
```

Here:

- `mu_c` is the spatial mean of channel `c`
- `v_c` is the spatial variance of channel `c`
- `epsilon` ensures numerical stability
- `gamma_c` and `beta_c` are generated for the current image

The output heads are initialized to zero, so QGIN initially behaves like standard instance normalization and gradually learns image-specific corrections during training.

<p align="center">
  <img src="Fig2_QGIN_Module.png" width="850" alt="Quantum-Generated Instance Normalization module"/>
</p>

### **3. Quantum Skip Coefficient Modulation**

Skip connections preserve boundaries and fine spatial details, but they may also transfer speckle noise, artifacts, and irrelevant background responses.

For the skip feature \(S_l\) at resolution level \(l\), QSCM combines:

- A pooled descriptor of the skip feature
- A pooled descriptor of the quantum-adapted bottleneck

The combined descriptor is projected into the VQC input dimension. Quantum measurements are then mapped to a channel-wise gate:

```math
g_l
=
\mathrm{tanh}
\left(
h_l
\left(
\mathrm{VQC}_l
\left(
f_l
\left(
\left[
\mathrm{GAP}(S_l),
\mathrm{GAP}(B)
\right]
\right)
\right)
\right)
\right)
```

The skip feature is recalibrated through residual modulation:

```math
\widetilde{S}_l
=
S_l
\odot
\left(1+g_l\right)
```

Because the gate values are bounded between -1 and 1, the multiplier \(1+g_l\) remains between 0 and 2:

- Negative values suppress irrelevant channels
- Values near zero preserve the original feature
- Positive values amplify informative channels

Independent QSCM branches are applied to the 64-, 128-, and 256-channel skip features.

<p align="center">
  <img src="Fig3_QSCM_Module.png" width="850" alt="Quantum Skip Coefficient Modulation module"/>
</p>

### **4. Quantum Spline-Adaptive Gate**

QSAG controls the overall strength of the bottleneck representation before it is passed to the decoder and the QSCM branches.

First, Global Average Pooling summarizes the bottleneck. A **Kolmogorov–Arnold Network** then performs nonlinear compression before the descriptor enters a compact VQC.

The KAN layer aggregates learnable spline functions placed on network edges:

```math
h_j^{(l)}
=
\sum_{i=1}^{m_{l-1}}
\phi_{i,j}^{(l)}
\left(
h_i^{(l-1)}
\right)
```

The resulting scalar gate is generated as:

```math
a
=
\mathrm{tanh}
\left(
h_{\mathrm{QSAG}}
\left(
\mathrm{VQC}_{\mathrm{QSAG}}
\left(
\mathrm{KAN}
\left(
\mathrm{GAP}(X)
\right)
\right)
\right)
\right)
```

The complete bottleneck is modulated using a residual gate:

```math
\widetilde{X}
=
X\left(1+a\right)
```

The scalar \(a\) is shared across all channels and spatial positions of the current image. Therefore, QSAG can attenuate, preserve, or amplify the entire semantic representation without changing its internal spatial organization.

<p align="center">
  <img src="Fig4_QSAG_Module.png" width="850" alt="Quantum Spline-Adaptive Gate module"/>
</p>

## **Quantum Circuit Design**

The quantum modules use compact VQCs rather than attempting to encode full image feature maps.

Depending on the module, the circuits use:

- Amplitude encoding or QAOA-inspired embedding
- Trainable single-qubit rotations
- Strongly Entangling Layers or Basic Entangler Layers
- Pauli-Z measurements
- Classical linear heads that map quantum measurements to adaptive coefficients

A general VQC output can be written as:

```math
q_k
=
\left\langle
\psi(\theta,x)
\right|
Z_k
\left|
\psi(\theta,x)
\right\rangle
```

where \(q_k\) is the measured expectation value of qubit \(k\), \(x\) is the compact image descriptor, and \(\theta\) contains trainable quantum-circuit parameters.

All classical weights and quantum parameters are optimized jointly through end-to-end training.

## **Datasets**

Experiments are conducted on three public breast ultrasound datasets. Only images containing breast lesions are used for the segmentation experiments.

| Dataset | Description | Images used for lesion segmentation |
|---|---|---:|
| **BUSI** | 780 images from 600 patients: 437 benign, 210 malignant, and 133 normal images | 647 lesion images |
| **BUS** | Images collected using a Siemens ACUSON Sequoia C512 system: 110 benign and 53 malignant cases | 163 lesion images |
| **BLUI** | Breast lesion ultrasound images: 123 malignant and 109 benign cases | 232 lesion images |

Each lesion image is accompanied by a manually annotated mask used as segmentation ground truth.

## **Implementation Details**

| Configuration | Value |
|---|---|
| Framework | PyTorch |
| Input resolution | 224 x 224 |
| Optimizer | AdamW |
| Initial learning rate | 1e-4 |
| Batch size | 8 |
| Training epochs | 80 |
| Evaluation strategy | Five-fold cross-validation |
| Reported values | Mean +/- standard deviation |
| Training objective | Binary Cross-Entropy + Dice Loss |

The training loss is an equally weighted combination of Binary Cross-Entropy and Dice loss:

```math
\mathcal{L}
=
0.5\mathcal{L}_{\mathrm{BCE}}
+
0.5\mathcal{L}_{\mathrm{Dice}}
```

Data augmentation includes:

- Random horizontal flipping
- Random rotations
- Gamma intensity transformations
- Blur augmentation
- Gaussian noise injection
- Elastic deformation with affine perturbations

## **Experimental Results**

QAC-UNet is evaluated against representative CNN-based, Transformer-based, and hybrid segmentation architectures, including:

- U-Net
- R34 DeepLabV3
- TransUNet
- SwinUNet
- AAU-Net
- HCT-Net
- HAU-Net

### **QAC-UNet Results Across Datasets**

| Dataset | Dice | Accuracy | IoU / Jaccard | Recall | Precision | HD95 |
|---|---:|---:|---:|---:|---:|---:|
| **BUSI** | **0.9392 +/- 0.0146** | **0.9685 +/- 0.0130** | **0.8856 +/- 0.0289** | **0.8912 +/- 0.0350** | **0.8824 +/- 0.0400** | **9.1300 +/- 3.4200** |
| **BUS** | **0.9532 +/- 0.0400** | **0.9704 +/- 0.0510** | **0.9362 +/- 0.0620** | **0.9023 +/- 0.0560** | **0.9057 +/- 0.0400** | **5.1200 +/- 2.8900** |
| **BLUI** | **0.9645 +/- 0.0602** | **0.9752 +/- 0.0310** | **0.9462 +/- 0.0350** | **0.9023 +/- 0.0340** | **0.9154 +/- 0.0610** | **3.7300 +/- 1.1700** |

The paper reports the highest Dice and IoU/Jaccard values across all three datasets. It also reports the lowest HD95 on BUSI and BLUI, while HAU-Net obtains the lowest HD95 on BUS.

## **Comparison with Previous Methods**

<details>
<summary><strong>BUSI comparison</strong></summary>

| Method | Dice | Accuracy | IoU / Jaccard | Recall | Precision | HD95 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.7713 +/- 0.0565 | 0.9538 +/- 0.0092 | 0.6370 +/- 0.0712 | 0.7359 +/- 0.0797 | 0.8263 +/- 0.0316 | 23.7013 +/- 1.9534 |
| R34 DeepLabV3 | 0.8007 +/- 0.0447 | 0.9602 +/- 0.0059 | 0.6752 +/- 0.0593 | 0.7859 +/- 0.0676 | 0.8293 +/- 0.0253 | 18.2490 +/- 1.4572 |
| TransUNet | 0.8118 +/- 0.0227 | 0.9628 +/- 0.0051 | 0.6992 +/- 0.0303 | 0.8185 +/- 0.0389 | 0.8251 +/- 0.0301 | 35.2600 +/- 3.3700 |
| SwinUNet | 0.8022 +/- 0.0198 | 0.9598 +/- 0.0065 | 0.7169 +/- 0.0244 | 0.8015 +/- 0.0450 | 0.8041 +/- 0.0203 | 14.2500 +/- 2.6500 |
| AAU-Net | 0.8036 +/- 0.0313 | 0.9609 +/- 0.0036 | 0.6909 +/- 0.0425 | 0.8050 +/- 0.0429 | 0.8283 +/- 0.0159 | 17.7480 +/- 2.9848 |
| HCT-Net | 0.8200 +/- 0.0189 | 0.9694 +/- 0.0048 | 0.7184 +/- 0.0252 | 0.8214 +/- 0.0203 | 0.8324 +/- 0.0186 | Not reported |
| HAU-Net | 0.8311 +/- 0.0207 | 0.9680 +/- 0.0016 | 0.7526 +/- 0.0208 | Not reported | 0.8608 +/- 0.0252 | 10.6700 +/- 2.4400 |
| **QAC-UNet** | **0.9392 +/- 0.0146** | **0.9685 +/- 0.0130** | **0.8856 +/- 0.0289** | **0.8912 +/- 0.0350** | **0.8824 +/- 0.0400** | **9.1300 +/- 3.4200** |

</details>

<details>
<summary><strong>BUS comparison</strong></summary>

| Method | Dice | Accuracy | IoU / Jaccard | Recall | Precision | HD95 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.8144 +/- 0.0775 | 0.9670 +/- 0.0334 | 0.7050 +/- 0.0916 | 0.7843 +/- 0.0894 | 0.8819 +/- 0.0443 | 14.2727 +/- 5.2269 |
| R34 DeepLabV3 | 0.8141 +/- 0.0518 | 0.9832 +/- 0.0028 | 0.6968 +/- 0.0692 | 0.7919 +/- 0.0676 | 0.8604 +/- 0.0246 | 24.0400 +/- 6.7200 |
| TransUNet | 0.8396 +/- 0.0326 | 0.9860 +/- 0.0046 | 0.7333 +/- 0.0422 | 0.8264 +/- 0.0278 | 0.8457 +/- 0.0400 | 21.1400 +/- 8.7900 |
| SwinUNet | 0.8551 +/- 0.0111 | 0.9862 +/- 0.0041 | 0.7584 +/- 0.0176 | 0.8623 +/- 0.0312 | 0.8445 +/- 0.0231 | 15.2800 +/- 1.6300 |
| AAU-Net | 0.8628 +/- 0.0363 | 0.9828 +/- 0.0122 | 0.7724 +/- 0.0466 | 0.8501 +/- 0.0539 | 0.9017 +/- 0.0316 | 8.3440 +/- 3.7579 |
| HCT-Net | 0.8413 +/- 0.0202 | 0.9849 +/- 0.0036 | 0.7383 +/- 0.0278 | 0.8319 +/- 0.0312 | 0.8850 +/- 0.0306 | Not reported |
| HAU-Net | 0.8873 +/- 0.0211 | 0.9903 +/- 0.0032 | 0.8122 +/- 0.0230 | Not reported | 0.8868 +/- 0.0225 | **3.6400 +/- 2.2600** |
| **QAC-UNet** | **0.9532 +/- 0.0400** | **0.9704 +/- 0.0510** | **0.9362 +/- 0.0620** | **0.9023 +/- 0.0560** | **0.9057 +/- 0.0400** | 5.1200 +/- 2.8900 |

</details>

<details>
<summary><strong>BLUI comparison</strong></summary>

| Method | Dice | Accuracy | IoU / Jaccard | Recall | Precision | HD95 |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | 0.8621 +/- 0.0199 | 0.9582 +/- 0.0040 | 0.7596 +/- 0.0292 | 0.8535 +/- 0.0181 | 0.8741 +/- 0.0327 | 15.7890 +/- 2.5638 |
| R34 DeepLabV3 | 0.8501 +/- 0.0859 | 0.9702 +/- 0.0356 | 0.7625 +/- 0.1004 | 0.8361 +/- 0.1015 | 0.9035 +/- 0.0527 | 9.7968 +/- 2.6393 |
| TransUNet | 0.8483 +/- 0.0177 | 0.9542 +/- 0.0029 | 0.7377 +/- 0.0259 | 0.8390 +/- 0.0222 | 0.8632 +/- 0.0377 | 16.7793 +/- 2.0218 |
| SwinUNet | 0.8617 +/- 0.0189 | 0.9586 +/- 0.0023 | 0.7583 +/- 0.0284 | 0.8471 +/- 0.0294 | 0.8808 +/- 0.0251 | 11.6144 +/- 0.9507 |
| AAU-Net | 0.8795 +/- 0.0135 | 0.9650 +/- 0.0034 | 0.7909 +/- 0.0186 | 0.8685 +/- 0.0183 | 0.8989 +/- 0.0271 | 10.2260 +/- 1.9335 |
| HCT-Net | 0.8551 +/- 0.0142 | 0.9563 +/- 0.0015 | 0.7479 +/- 0.0216 | 0.8463 +/- 0.0254 | 0.8685 +/- 0.0343 | Not reported |
| HAU-Net | 0.8948 +/- 0.0044 | 0.9696 +/- 0.0042 | 0.8212 +/- 0.0085 | Not reported | 0.8993 +/- 0.0115 | 5.3800 +/- 0.6600 |
| **QAC-UNet** | **0.9645 +/- 0.0602** | **0.9752 +/- 0.0310** | **0.9462 +/- 0.0350** | **0.9023 +/- 0.0340** | **0.9154 +/- 0.0610** | **3.7300 +/- 1.1700** |

</details>

## **Why QAC-UNet?**

QAC-UNet is designed to balance:

- Local feature extraction through CNNs
- Global-context modeling through Transformer blocks
- Multi-scale representation through ASPP
- Image-specific adaptation through quantum-generated coefficients
- Nonlinear descriptor compression through KAN
- Quantum-resource efficiency by avoiding full feature-map encoding
- Accurate region overlap and boundary localization

The core idea is not to replace the classical segmentation network with a quantum model. Instead, compact quantum circuits act as adaptive controllers that determine how selected classical features should be normalized, suppressed, preserved, or amplified for each image.

## **Paper**

**Title:** QAC U-NET: Quantum Adaptive Coefficient UNET for Breast Ultrasound Segmentation

**Authors:**

- Nhat-Long Nguyen
- Viet-Loi Dinh
- Trong-Nghia Pham

**Affiliation:** Faculty of Information Technology, University of Science, Vietnam National University, Ho Chi Minh City, Vietnam

## **Citation**

The citation information will be updated after the paper is officially published.

```bibtex
@inproceedings{nguyen2026qacunet,
  title     = {QAC U-NET: Quantum Adaptive Coefficient UNET for Breast Ultrasound Segmentation},
  author    = {Nguyen, Nhat-Long and Dinh, Viet-Loi and Pham, Trong-Nghia},
  booktitle = {Proceedings of ACIVS 2026},
  year      = {2026},
  note      = {Under review}
}
```

## **Acknowledgments**

This research is supported by research funding from the **Faculty of Information Technology, University of Science, Vietnam National University – Ho Chi Minh City**.
