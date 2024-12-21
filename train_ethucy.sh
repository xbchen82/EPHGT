# This .sh script is used in the linux system to run the train.py file.
for i in 0 1 2 3 4
do
    python train.py --test_set $i --num_epochs 1000 --x_encoder_layers 3 --eta_min 1e-6  --batch_size 32 --learning_rate 5e-4  --randomRotate True --final_mode 20 --neighbor_thred 10 --using_cuda True --clip 1 --pass_time 2 --ifGaussian False --SR False --input_offset True
done