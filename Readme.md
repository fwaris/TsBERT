### TsBERT

BERT model defined in TorchSharp with functionality to load pre-trained BERT weights from checkpoints made available by Google.

This model produces the pooled / encoded output - a single value per input sequence. It is meant to be used as a base layer for models for common NLP tasks e.g. text classification.

The code in this package is a re-packaging of the code shown in [this notebook](https://github.com/fwaris/BertTorchSharp/blob/master/saved_output.ipynb). 
Please consult the notebook (and associated repo) for a text classification example.

Also see BertFineTuneExample.fsx script file in the repo for the same example in a script file

(updated to torchsharp 0.98.1)




