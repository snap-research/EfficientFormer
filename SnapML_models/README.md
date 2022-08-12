# Using EfficientFormer
## Training EfficientFormer
In order to train a SnapML Compatible EfficientFormer, you should be able to swap out the models/efficientfomer.py with the content from SnapML_models/EfficientFormer.py and run main.py as discussed in the main README

## Converting EfficientFormer
Once you have trained the EfficientFormer, you should be able to convert it to a format that can be imported into Lens Studio. To do so, call the conversion script as follows:

```sh
python path/to/model.pth
```