# Clinical 3DMA Exporter

## Data Export
Clinical 3DMA Exporter can export the following types of data:

* ```3mc```. 3MC is the proprietary file format created by STT and used across all 3DMA.products.
* ```biomechanics```. CSV files containing biomechanical data
* ```doc```. DOCX file containing a rich gait report
* ```video```.
* ```video3D```. AVI file containing the 3D view of a motion capture
* ```strides```. CSV files with data of the strides
* ```3dt```.
* ```trc```. TRC file containing marker trajectories
* ```raw```. TXT file containing marker trajectories
* ```events```. TXT file containing event times
* ```c3d```

## Usage

```
env/Scripts/activate.bat
```

```
pip install -r requirements.txt
```

```
python 3dma_exporter.py
```