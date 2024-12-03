export PYTHONPATH=/root/OEST:$PYTHONPATH
export LANG=en_US.UTF-8

s_values=(s0 s1 s2)
for s in "${s_values[@]}";
do
    echo "Running with $s"

    python oest/cifar100/main.py -g 0 --todo train -lr 1e-3 --alpha 0.2 --beta 10 -s 1 --affix seed1_${s}_0.2_10 --load_checkpoint ./cifar100_resnet18_32x32/${s}/best.ckpt 
done