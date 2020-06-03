* [Diagnosing Alzheimer's Disease](#diagnosing-alzheimers-disease)<br>
* [The Oasis Brains Datasets](#the-oasis-brains-datasets)<br>
* [Identifying Neurodegeneration with a Neural Network](#identifying-neurodegeneration-with-a-neural-network)<br>
* [Predicting Alzheimer's in Cognitively Normal Subjects](#predicting-alzheimers-in-cognitively-normal-subjects)

# Diagnosing Alzheimer's Disease

**Alzheimer's disease** is the most common form of dementia, effecting almost 12% of americans over the age of 60 and over 50 million people worldwide. Alzheimer's is best characterized by memory problems, movement difficulties, anosmia, and impaired reasoning.

Currently Alzheimer's disease is diagnosed through a physician by a combination of reported behavioral changes, patient history, and a mental status exam. As behavioral symptoms can overlap heavily with other neurodegenerative disorders, such as vascular dementia, this leaves ample room for misdiagnosis. Currently, Alzheimer's disease can only be *definitively*  diagnosed postmortem. This is because the hallmarks of Alzheimer's, proteinaceous changes that include amyloid plaques and neurofibrillary tangles, can only be oberved with brain tissue itself. 

The trammels of diagnosis are a major barrier to the development of Alzheimer's therapies. Assessing the efficacy of a drug in clinical trials is hindered by a potentially heterogenous patient population. Developing more accurate means of identifying the disease in living patients will facilitate medical advances in the field. 

<p align="center">
  <b>MRI of Cognitively Normal Patient</b><br>
  <img src="https://github.com/GMattheisen/predicting_Alzheimers_from_MRI/blob/master/brain_COG_NORM.jpg"><br><br>
  <b>MRI of AD patient</b><br>
  <img src="https://github.com/GMattheisen/predicting_Alzheimers_from_MRI/blob/master/brain_ALZ.jpg">
</p>

# The Oasis Brains Datasets

Data used in this report are taken with permission from the [OASIS Brains Datasets](https://www.oasis-brains.org/#data). 

**OASIS-1 Summary**: This set consists of a cross-sectional collection of 416 subjects aged 18 to 96. For each subject, 3 or 4 individual T1-weighted MRI scans obtained in single scan sessions are included. The subjects are all right-handed and include both men and women. 100 of the included subjects over the age of 60 have been clinically diagnosed with very mild to moderate Alzheimer’s disease (AD). 

**OASIS-3 Summary**: This set is a retrospective compilation of data for >1000 participants that were collected across several ongoing projects through the WUSTL Knight ADRC over the course of 30 years. Participants include 609 cognitively normal adults and 489 individuals at various stages of cognitive decline ranging in age from 42-95yrs. All participants were assigned a new random identifier and all dates were removed and normalized to reflect days from entry into study. The dataset contains over 2000 MR sessions which include T1w, T2w, FLAIR, ASL, SWI, time of flight, resting-state BOLD, and DTI sequences. Many of the MR sessions are accompanied by volumetric segmentation files produced through Freesurfer processing. PET imaging from 3 different tracers, PIB, AV45, and FDG, totaling over 1500 raw imaging scans and the accompanying post-processed files from the Pet Unified Pipeline (PUP) are also available in OASIS-3. 

Further information on data import and wrangling can be found in the individual project files.

# [Identifying Neurodegeneration with a Neural Network](https://github.com/GMattheisen/predicting_Alzheimers_from_MRI/blob/master/predicting_Alzheimers_from_MRI.py)

In this project, I trained neural networks and random forest classifiers to distinguish the brains of cognitively normal patients from those diagnosed with Alzheimer's disease. 

I applied PCA to masked transverse-orientation MRI images from the OASIS-2 dataset in order to build a neural network that could discriminate healthy brains from brains of patients diagnosed with Alzheimer's disease with **94.6%** accuracy. This out-performed the predictive accuracy of a random forest classifier analysis of the derived anatomical measures and demographic data from the same patients by **~8%**. 

Data show the power of neural networks for image recognition and the immense potential of these machine learning methods for neurodegeneration diagnoses. 

# [Predicting Alzheimer's in Cognitively Normal Subjects](https://github.com/GMattheisen/predicting_Alzheimers_from_MRI/blob/master/predicting_AD_in_CogNorm.py)

Studies show that Alzheimer's-associated neurodegeneration can occur almost a decade before cognitive symptoms emerge. During this preclinical stage, patient's have normal cognitive function, but amyloid plaque deposits are already accumulating in brain tissue, leading to neuronal death. While early degeneration begins in the hippocampus, the area of the brain associated with memory formation, in late-stage Alzheimer's disease, degeneration is widespread.

The OASIS-3 dataset includes extensive demographic data and derived anatomical measures from healthy patients and those with Alzheimer's disease. I used a random forest classifier to distinguish the two groups with **93%** accuracy. The dataset contained 145 patients who were classified as having Alzheimer's disease *during* the study. Derived anatomical measures before and after the Alzheimer's diagnosis were included for these subjects. 

I then asked if the random forest classifier could predict the development of Alzheimer's disease in these patients before their clinical diagnosis. The random forest classifier was subsequently trained on data from patients whose diagnosis did not changed during the study, as well as only the data from the patients whose diagnosis did change, after the diagnosis of Alzheimer's disease was made. I then fed back the data from these 145 patients later diagnosised as having Alzheimer's disease back into the model. The model assigned the diagnosis of Alzheimer's disease to these cognitively normal patients in **~70%** of cases. This shows that the random forest classifier could predict the diagnosis of Alzehimer's disease with high accuracy in patients described as cognitively normal by a physician. 

When therapies for Alzheimer's disease are developed, early diagnosis methods such as this will be integral to securing the best patient outcomes. 

*Information supplied from the [NIH National Institute on Aging](https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet)*
