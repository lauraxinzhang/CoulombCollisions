
ps -p $$
ORDER=0
#source activate coulomb
echo $ORDER
python convTests.py -w MEM 4 3 0.1 16 $ORDER
