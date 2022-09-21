
```py
!pip install -U pip setuptools
!pip install google-colab

from google.colab import drive
drive.mount('/content/drive')
```

```py
!pwd
!ls -l .
!ls -l /content/drive/MyDrive/colab/ddsp_pytorch/input

!tar -xvf /content/drive/MyDrive/colab/ddsp_pytorch/input/ddsp.tar.gz

!ls -l .
```

```py
!pip install -r requirements.txt
```

```py
!export DATA_DIR=/content/drive/MyDrive/colab/ddsp_pytorch/input/ddsp_preprocessed
!export TRAIN_DIR=/content/drive/MyDrive/colab/ddsp_pytorch/output

!mkdir -p $TRAIN_DIR
!python train.py
```