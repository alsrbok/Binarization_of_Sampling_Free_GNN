#xnor_net_sign
    #ogbn_products
    python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --seed 0 --model xnor_net_sign \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50000 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 0 --warmup-stage -1


#xnor_case1_sign
    #ogbn_products
nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --seed 0 --model xnor_case1_sign \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50000 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 0 --warmup-stage -1 > xnor_case1_sign_products.out

    #cora
nohup python -u sagn.py \
    --dataset cora \
    --gpu 0 --aggr-gpu -1 \
    --model xnor_case1_sign --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50 \
    --num-hidden 64 --dropout 0.5 \
    --input-drop 0. --K 9 \
    --mlp-layer 2 --use-norm \
    --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sign_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 0 --aggr-gpu -1 \
    --model xnor_case1_sign --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --input-drop 0.0 --use-norm \
    --K 2 --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sign_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 1 --aggr-gpu -1 \
    --model xnor_case1_sign --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --input-drop 0.0 \
    --K 3 --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sign_reddit.out &





#3by4_xnor_case1_sign
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --seed 3 --model 3by4_xnor_case1_sign \
    --num-runs 7 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50000 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 0 --warmup-stage -1 > 3by4_xnor_case1_sign_products.out &

    #cora
nohup python -u sagn.py \
    --dataset cora \
    --gpu 0 --aggr-gpu -1 \
    --model 3by4_xnor_case1_sign --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50 \
    --num-hidden 64 --dropout 0.5 \
    --input-drop 0. --K 9 \
    --mlp-layer 2 --use-norm \
    --weight-decay 0 \
    --warmup-stage -1 > 3by4_xnor_case1_sign_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 0 --aggr-gpu -1 \
    --model 3by4_xnor_case1_sign --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --input-drop 0.0 --use-norm \
    --K 2 --weight-decay 0 \
    --warmup-stage -1 > 3by4_xnor_case1_sign_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 1 --aggr-gpu -1 \
    --model 3by4_xnor_case1_sign --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --input-drop 0.0 \
    --K 3 --weight-decay 0 \
    --warmup-stage -1 > 3by4_xnor_case1_sign_reddit.out &


# hce_sign_binarize 
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 1 --aggr-gpu -1 \
    --seed 0 --model HCE_sign_xnor_case1 \
    --num-runs 5 --epoch-setting 1000 \
    --lr 0.001 --batch-size 600 \
    --eval-batch-size 600 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 0 \
    --warmup-stage -1 > HCE_sign_xnor_case1_products.out &

    #cora
    nohup python -u sagn.py \
    --dataset cora \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sign_xnor_case1 \
    --acc-loss acc --threshold 0.9 \
    --eval-every 10 --num-runs 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 50 --num-hidden 64 \
    --dropout 0.5 --input-drop 0.\
    --K 9 --mlp-layer 2 \
    --use-norm --weight-decay 0 \
    --warmup-stage -1 >  HCE_sign_xnor_case1_cora.out &

    #flickr
    nohup python -u sagn.py \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sign_xnor_case1\
    --dataset flickr --zero-inits \
    --inductive --eval-every 10 \
    --num-runs 10 --epoch-setting 200 \
    --threshold 0.5 --lr 0.001 \
    --batch-size 128 --eval-batch-size 128 \
    --num-hidden 512 --dropout 0.7 \
    --input-drop 0.0 --use-norm \
    --K 2 --weight-decay 0 \
    --warmup-stage -1 > HCE_sign_xnor_case1_flickr.out &

    #reddit : not yet..
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sign_xnor_case1 \
    --inductive --zero-inits \
    --threshold 0.9 0.9 --eval-every 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 10000 --eval-batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7  --input-drop 0.0 \
    --K 3 --weight-decay 0 \
    --warmup-stage -1 > HCE_sign_xnor_case1_reddit.out &