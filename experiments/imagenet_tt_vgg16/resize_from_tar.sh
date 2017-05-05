#!/bin/bash
# sudo apt-get update
# sudo apt-get install imagemagick
# Start in folder with original .tar files
#Before launch set PACKEDTRAINDIR and DATA_OUT
# Variables
PACKEDTRAINDIR=/home/ubuntu/tmp/train_packed
DATA_OUT=/home/ubuntu/imagenet_mimic
TRAINDIR=$DATA_OUT/images/train
VALDIR=$DATA_OUT/images/val

mkdir $DATA_OUT
mkdir $DATA_OUT/images
# Unpack main train archive
mkdir $PACKEDTRAINDIR
tar -xvf ILSVRC2012_img_train.tar -C $PACKEDTRAINDIR

# Unpack & resize nested train archives
mkdir $TRAINDIR

for NAME in $PACKEDTRAINDIR/*.tar; do
  INDEX=$(basename $NAME .tar)
  echo $INDEX
  if test -d $TRAINDIR/$INDEX; then
   echo "Folder "$TRAINDIR/$INDEX" exists"
  else
  mkdir $TRAINDIR/$INDEX
  tar -xf $PACKEDTRAINDIR/$INDEX.tar -C $TRAINDIR/$INDEX
  # Resize to height to 256, preserving the aspect ratio
  mogrify -resize 256x256^   "$TRAINDIR/$INDEX/*.JPEG"
  fi
done

# Validation
mkdir $VALDIR
tar -xf ILSVRC2012_img_val.tar -C $VALDIR
mogrify -resize 256x256^   "$VALDIR/*.JPEG"


# CMYK -> RGB. Important
mogrify -colorspace rgb $TRAINDIR/n03062245/n03062245_4620.JPEG
mogrify -colorspace rgb $TRAINDIR/n04264628/n04264628_27969.JPEG
mogrify -colorspace rgb $TRAINDIR/n03961711/n03961711_5286.JPEG
mogrify -colorspace rgb $TRAINDIR/n01739381/n01739381_1309.JPEG
mogrify -colorspace rgb $TRAINDIR/n04258138/n04258138_17003.JPEG
mogrify -colorspace rgb $TRAINDIR/n03018349/n03018349_4028.JPEG
mogrify -colorspace rgb $TRAINDIR/n04336792/n04336792_7448.JPEG
mogrify -colorspace rgb $TRAINDIR/n02492035/n02492035_15739.JPEG
mogrify -colorspace rgb $TRAINDIR/n03544143/n03544143_17228.JPEG
mogrify -colorspace rgb $TRAINDIR/n03467068/n03467068_12171.JPEG
mogrify -colorspace rgb $TRAINDIR/n03633091/n03633091_5218.JPEG
mogrify -colorspace rgb $TRAINDIR/n02447366/n02447366_23489.JPEG
mogrify -colorspace rgb $TRAINDIR/n03347037/n03347037_9675.JPEG
mogrify -colorspace rgb $TRAINDIR/n02077923/n02077923_14822.JPEG
mogrify -colorspace rgb $TRAINDIR/n02747177/n02747177_10752.JPEG
mogrify -colorspace rgb $TRAINDIR/n04371774/n04371774_5854.JPEG
mogrify -colorspace rgb $TRAINDIR/n07583066/n07583066_647.JPEG
mogrify -colorspace rgb $TRAINDIR/n04596742/n04596742_4225.JPEG
mogrify -colorspace rgb $TRAINDIR/n13037406/n13037406_4650.JPEG
mogrify -colorspace rgb $TRAINDIR/n03529860/n03529860_11437.JPEG
mogrify -colorspace rgb $TRAINDIR/n03710637/n03710637_5125.JPEG
mogrify -colorspace rgb $TRAINDIR/n04033995/n04033995_2932.JPEG
mogrify -colorspace rgb $VALDIR/ILSVRC2012_val_00019877.JPEG
