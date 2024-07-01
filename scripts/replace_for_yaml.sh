set -e
sed "s/$2/$3/" -i $1
echo "repalced all the $2 to $3 in $1 ~"