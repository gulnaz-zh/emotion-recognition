# emotion-recognition

emotion_mlp.ipynb provides code for MLP evaluations based on preprocessed data. Before running it, several steps should be accomplished:
  - preprocess.py provides predictions of VGG16, 3D CNN for micro expression and fine-tuned 3D CNN models.
  - get ActionUnit (regression) values from https://github.com/TadasBaltrusaitis/OpenFace and preprocess it using ActionUnit_preprocess.ipynb file
  - vgg11_features.py provides image features from VGG11 network
 
 Furthermore, for fine-tuning purposes cnn3d_finetuning.py and vgg16_finetuning files are provided.
 
 Some pretrained models are provided in models folder. 3D CNN models are not uploaded because of GitHub memory restrictions.
 
 For any dataset evaluation, dataset should be arranged as shown in the dataset folder.

The code is for:

@article{lukac2023study,
  title={Study on emotion recognition bias in different regional groups},
  author={Lukac, Martin and Zhambulova, Gulnaz and Abdiyeva, Kamila and Lewis, Michael},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={8414},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
