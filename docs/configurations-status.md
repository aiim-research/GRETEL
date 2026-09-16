# Configurations
## Tested and working configurations:

### ADHD
* legacy/config-v2/ADHD_SVM-G2V_DCE.jsonc

### ASD
* legacy/config-v2/ASD_ASD-Custom_DCE.jsonc
* legacy/config-v2/ASD_ASD-Custom_DDBS.jsonc
* legacy/config-v2/ASD_ASD-Custom_OBS.jsonc
* legacy/config-v2/ASD_SVM-G2V_DCE.jsonc
* legacy/config-v2/ASD+TCRs+BBBP_GCN_DCE.jsonc

### BBBP
* legacy/config-v2/BBBP_GCN_DCE.jsonc
* legacy/config-v2/BBBP_GCN_MACCS.json
* legacy/config-v2/BBBP_KNN-MOL_DCE.json
* legacy/config-v2/BBBP_SVM-MOL_DCE.json

### TCR
* legacy/config-v2/TCR-128-32-0.2_GCN_DCE.jsonc
* legacy/config-v2/TCR-128-32-0.2_GCN_RSGG.jsonc
* legacy/config-v2/TCR-128-32-0.2_KNN-G2V_DCE.jsonc
* legacy/config-v2/TCR-128-32-0.2_SVM-G2V_DCE.jsonc
* legacy/config-v2/TCR-150-100-0.3_TCC_DCE.jsonc
* legacy/config-v2/TCR-150-100-0.3_TCC_iRand.jsonc
* legacy/config-v2/TCR-150-100-0.3_TCC_pRand.jsonc

### CF2 & CLEAR
* legacy/config-v2/explainers/cf2/ASD_ASD-Custom_CF2.jsonc
* legacy/config-v2/explainers/cf2/TCR-500-28.0.3_TCR-Custom_CF2.jsonc
* legacy/config-v2/explainers/clear/ASD_ASD-Custom_CLEAR.jsonc
* legacy/config-v2/explainers/clear/TCR-500-28-0.3_TCR-Custom_CLEAR.jsonc

## Tested and NOT working configurations:
* legacy/config-v2/TCR-500-64-0.4_GCN_RSGG.jsonc
* legacy/config-v2/TCR-500-64-0.4_GCN_GCounteRGAN.jsonc
* legacy/config-v2/BBBP_GCN_RSGG.json

## TODO
* legacy/config-v2/WIP/*

# Code

## Need a Test/Revision
* dataset/tree_cycles_fixed*
* dataset/manipulators/diameters.py
* dataset/manipulators/rank.py

* explainer/ensemble/*
* explainer/rl/*

* evaluation/*

* data_analysis/*