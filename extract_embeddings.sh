str=$1
mkdir $2
english="english"
french="french"
german="german"

if [ "$str" == "$english" ]
then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
	unzip glove.840B.300d.zip
	mv glove.840B.300d.txt $2
	sed -i '1 i\2196017 300' $2/glove.840B.300d.txt
	mv $2/glove.840B.300d.txt $2/embeddings
elif [ "$str" == "$french" ]
then
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec
	mv wiki.fr.vec $2/embeddings
elif [ "$str" == "$german" ]
then
	wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.nl.vec
	mv wiki.nl.vec $2/embeddings
fi
