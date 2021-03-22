# Build the client
src_folder="gym_gvgai/envs/gvgai/src"
build_folder="builtclient"
rm -rf $build_folder
mkdir -p $build_folder
find $src_folder -name "*.java" > sources.txt
javac -d $build_folder @sources.txt
# run builtclient
#java -classpath $build_folder tracks.singlePlayer.Test
