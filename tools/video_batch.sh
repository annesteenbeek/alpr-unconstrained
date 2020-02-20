video_folder=$1
script_folder=`dirname $0`


for file_name in `ls "$video_folder" | grep .mp4`;do
  echo $file_name
  python "$script_folder/video2frames.py" "$video_folder/$file_name"
done
for file_name in `ls "$video_folder" | grep .mov`;do
  echo $file_name
  python "$script_folder/video2frames.py" "$video_folder/$file_name"
done
