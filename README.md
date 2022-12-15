# emotion-recognition

emotion_mlp.ipynb provides code for MLP evaluations based on preprocessed data. Before running it, several steps should be accomplished:
  - preprocess.py provides predictions of VGG16, 3D CNN for micro expression and fine-tuned 3D CNN models.
  - get ActionUnit (regression) values from https://github.com/TadasBaltrusaitis/OpenFace and preprocess it using ActionUnit_preprocess.ipynb file
  - vgg11_features.py provides image features from VGG11 network
 
 Furthermore, for fine-tuning purposes cnn3d_finetuning.py and vgg16_finetuning files are provided.
 
 Some pretrained models are provided in models folder. 3D CNN models are not uploaded because of GitHub memory restrictions.
 
 For any dataset evaluation, dataset should be arranged as shown in the dataset folder.

