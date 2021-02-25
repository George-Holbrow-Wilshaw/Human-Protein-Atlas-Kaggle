# Human-Protein-Atlas-Kaggle

In this repository I am developing my entry to the Kaggle competition https://www.kaggle.com/c/hpa-single-cell-image-classification.

I am utilising a combination of Proposal Cluster Learning https://arxiv.org/pdf/1807.03342.pdf, and HPAcellsegmentation https://github.com/CellProfiling/HPA-Cell-Segmentation. 

My pipeline is planned as follows:


1) Generate masks using all 3 channels (nuclei, mitrochondria and endoplasmic reticulum) and the HPAcellsegmentation model.
2) Convert masks to bounding boxes using cv2. These are then taken and used as the regions of interest (ROIs) in the PCL model
3) Construct COCO style JSON annontations for the HPA dataset (needed for the PCL model)
4) Feed ROIs, annotations, and images into the PCL model.
5) Relate ground truth predictions back to original HPA masks
6) Post-processing for submission (RLE encode masks etc). 

This project is proving to be difficult for a number of reasons, not least my lack of an NVIDA GPU means I have to run everything on Google Colab and Kaggle, which makes making changes to PCL model codebase very slow to test.

The PCL model is represented by the below graph:

![Alt text](/misc/net.png)
