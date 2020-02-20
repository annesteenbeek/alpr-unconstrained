folder=$1


for i in `ls $folder`;do
  folder_anno=${folder}_anno/${i}
  bash run.sh \
    -i "$folder/${i}" \
    -o "$folder_anno" \
    -c "$folder_anno/results.csv";
done
