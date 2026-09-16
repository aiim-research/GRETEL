# Configurations
## Tested and working configurations:

### ADHD
* config/ADHD_SVM-G2V_DCE.jsonc

### ASD
* config/ASD_ASD-Custom_DCE.jsonc
* config/ASD_ASD-Custom_DDBS.jsonc
* config/ASD_ASD-Custom_OBS.jsonc
* config/ASD_SVM-G2V_DCE.jsonc
* config/ASD+TCRs+BBBP_GCN_DCE.jsonc

### BBBP
* config/BBBP_GCN_DCE.jsonc
* config/BBBP_GCN_MACCS.json
* config/BBBP_KNN-MOL_DCE.json
* config/BBBP_SVM-MOL_DCE.json

### TCR
* config/TCR-128-32-0.2_GCN_DCE.jsonc
* config/TCR-128-32-0.2_GCN_RSGG.jsonc
* config/TCR-128-32-0.2_KNN-G2V_DCE.jsonc
* config/TCR-128-32-0.2_SVM-G2V_DCE.jsonc
* config/TCR-150-100-0.3_TCC_DCE.jsonc
* config/TCR-150-100-0.3_TCC_iRand.jsonc
* config/TCR-150-100-0.3_TCC_pRand.jsonc

### CF2 & CLEAR
* config/explainers/cf2/ASD_ASD-Custom_CF2.jsonc
* config/explainers/cf2/TCR-500-28.0.3_TCR-Custom_CF2.jsonc
* config/explainers/clear/ASD_ASD-Custom_CLEAR.jsonc
* config/explainers/clear/TCR-500-28-0.3_TCR-Custom_CLEAR.jsonc

## Tested and NOT working configurations:
* config/TCR-500-64-0.4_GCN_RSGG.jsonc
* config/TCR-500-64-0.4_GCN_GCounteRGAN.jsonc
* config/BBBP_GCN_RSGG.json

## TODO
* config/WIP/*

# Code

## Need a Test/Revision
* dataset/tree_cycles_fixed*
* dataset/manipulators/diameters.py
* dataset/manipulators/rank.py

* explainer/ensemble/*
* explainer/rl/*

* evaluation/*

* data_analysis/*