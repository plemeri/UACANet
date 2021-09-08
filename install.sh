curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=17Cs2JhKOKwt4usiAYJVJMnXfyZWySn3s" -o data.zip

curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1C5ag5X_gKR1IHW6fVAHdMggu7ilU1XbC" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1C5ag5X_gKR1IHW6fVAHdMggu7ilU1XbC" -o snapshots.zip

mkdir data
mkdir snapshots
unzip data.zip -d data
unzip snapshots.zip -d snapshots
rm data.zip
rm snapshots.zip
rm cookie
clear
