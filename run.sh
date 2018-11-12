arch=(gens)
learn=(s)

n=500

if [ "$#" -eq 1 ]; then
  n=$1
fi

function run_drive
{
  (./godrive r saved/$4.mdl $1 $2 $n $3 data/$5.npy | tee $4.txt) && \
    (./godrive s saved/$4.mdl $1 $3 data/$5.npy | tee -a $4.txt)
}

for a in "${arch[@]}"; do
  for l in "${learn[@]}"; do
    for i in `seq 5 8`; do
      b=$(($(calc "2^${i} - 1")))
      mdl_name="${a}_${l}_${b}_${n}"
      data_name="train_${i}"
      echo "Running arch=$a learn=$l max=$b res=$i mdl=$mdl_name data=$data_name n=$n"
      run_drive ${a} ${l} ${b} ${mdl_name} ${data_name}
      if [ "$i" -ne 1 ]; then
        echo "Running equalized version"
        run_drive ${a} ${l} ${b} ${mdl_name}_eq ${data_name}_eq
      fi
    done
  done
done
