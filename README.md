# National-Scale Predictive Mapping of Zostera marina in Turbid UK Waters Using Sentinel-2 Imagery and Ensemble Machine Learning

## Instructions for viewing model predictions

The model probability maps are placed in the `model-predictions` folder and the threshold classifications are in `threshold-classifications`. The threshold classification maps are classifications of the probability maps with a threshold applied. For Random Forest, this was 0.27, and for XGBoost, it was 0.24.

The maps are in GeoTIFF file format and are divided into Sentinel-2 tiles (MGRS grid). They can be loaded into ArcGIS Pro or other GIS software by dragging and dropping the files to the map display. The script in the `arcgis` folder can be used to load the model predictions with a specific colormap, produce the threshold classifications and calculate the bed areas.
