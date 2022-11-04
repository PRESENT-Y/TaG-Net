## Usage: Vessel Completion
### Centerline Completion 
We complete the centerline based on the labeled vascular graph (output of the TaG-Net).

- First, we generate the connection pairs to connect the interrupted segments.

```python
python ./VesselCompletion/gen_connection_pairs.py
```
- visualize

```python
python ./VesselCompletion/vis_labeled_cl_graph.py
```
- Then, we search the connection path to complete the centerline.

```python
python ./VesselCompletion/gen_connection_path.py
```
- visualize

```python
python ./VesselCompletion/vis_connection_path.py
```

### Adhesion Removal
We remove the adhesion between segments with different labels based on the labeled vascular graph (output of the TaG-Net)

```python
python ./VesselCompletion/gen_adhesion_removal.py
```
- visualize

```python
python ./VesselCompletion/vis_adhesion_removal.py
```