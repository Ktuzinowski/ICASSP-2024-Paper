#!/bin/bash
mkdir imagenet

mkdir imagenet/train && mv ILSVRC2012_img_train.tar imagenet/train/ && cd imagenet/train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar

find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAf "${NME%.tar}"; rm -AME}"; done

cd ../..

mkdir imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar

wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
