# An-Automated-Explainable-Pathomics-Framework-for-Tertiary-Lymphoid-Structures
An automated computational pathology framework was proposed to discover the correlations between the spatial, architectural, and maturational heterogeneity of Tertiary Lymphoid Structures (TLSs) and tumor microenvironment in Pancreatic Ductal Adenocarcinoma (PDAC). Based on 7-color multiplex immunofluorescence (mIF) whole-slide images, this framework is further deployed for automated TLS segmentation, maturation grading, unsupervised phenotyping, and generating an explainable Random Survival Forest (RSF)-based risk score for patient prognostic stratification.

The goal of this repository is:
- to help researchers to reproduce the TLS pathomics framework (including mIF preprocessing, cell segmentation, TLS boundary delineation, and pathomics feature extraction) and expand it for immune microenvironment research in other solid tumors.
- to help researchers build an end-to-end, explainable machine learning workflow to predict patient overall survival (OS) by capturing biologically interpretable TLS morphology and spatial organization, assisting in standardized prognostic assessment beyond conventional clinicopathological factors.
- to provide custom Python scripts, QuPath Groovy scripts, and the trained RSF prognostic model for the migration of downstream tasks such as quantitative profiling of spatial interactions and immune cell architectures in multiplex imaging.

## Installation
```
conda create -n tls_pathomics python=3.9
conda activate tls_pathomics
pip install -r requirements.txt

git clone https://github.com/StandWisdom/An-Automated-Explainable-Pathomics-Framework-for-Tertiary-Lymphoid-Structures.git
cd An-Automated-Explainable-Pathomics-Framework-for-Tertiary-Lymphoid-Structures
```
## Quick Start
To run the full pipeline, you only need to interact with `main.py`.

Open `main.py` and Modify the read_path variable to point to your `.czi` file directory. Modify the pmax and gpu_id if necessary.
```
# Inside main.py
pmax = 16400  # Max pixel intensity threshold
gpu_id = '0'  # Select GPU ID
read_path = '/path/to/your/czi/directory' 
```
Run the script
```
python main.py
```

## Pipeline Detailed Steps
1. WSI Preprocessing (P1-P3): Reads .czi files using slideio, performs histogram equalization, extracts thumbnails, and automatically detects candidate TLS regions using morphological operations.
2. AI-driven Single Cell Segmentation (P4-P5): Uses Cellpose on the DAPI channel to detect nuclei, followed by a Watershed algorithm to segment the cytoplasm.
3. TLS Identification & Maturation Grading (P6-P8):
   - P6: Identifies Lymphocyte Aggregates (LAs) by overlaying CD20 and CD3 expression on segmented cells. 
   - P7: Identifies Primary Follicle-like TLS (FLI-TLS) using CD21 contiguous areas.
   - P8: Identifies Secondary Follicle-like TLS (FLII-TLS) using CD23 expression.
4. QuPath & Visualization (P9-P10): Maps the classified TLSs back to the whole-slide coordinates and generates QuPath-compatible `.json` files.
5. Pathomics Feature Extraction (P11): Extracts 39 features, including morphology (area, perimeter, diameter), cellular density (CD20/CD3 ratio), and maturation status. Saves as `extract_result.csv`.
6. Tumor Proximity & Spatial Analysis (P12-P14): Calculates the interaction between TLSs and the surrounding tumor microenvironment using PanCK and Ki-67 channels at different radius ratios.
