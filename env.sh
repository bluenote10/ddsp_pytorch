export TRAIN_DIR=$(pwd)/runs
export DATA_DIR=${HOME}/gdrive/colab/ddsp_pytorch/input/ddsp_preprocessed

echo "TRAIN_DIR=$TRAIN_DIR"
echo "DATA_DIR=$DATA_DIR"

workon ddsp_pytorch
