source=$1
name=`basename $source`
folder=`dirname $source`

mv "$1.pb" "$folder/saved_model.pb"
mkdir "$folder/variables"
mv "$1.ckpt.data-00000-of-00001" "$folder/variables/variables.data-00000-of-00001"
mv "$1.ckpt.index" "$folder/variables/variables.index"
