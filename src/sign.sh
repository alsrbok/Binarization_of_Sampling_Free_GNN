# sign_basic
    #ogbn_products
    python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --seed 0 --model sign \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50000 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 0 \
    --warmup-stage -1

    #cora
nohup python -u sagn.py \
    --dataset cora \
    --gpu 2 --aggr-gpu -1 \
    --model sign --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50 \
    --num-hidden 64 --dropout 0.5 \
    --input-drop 0. --K 9 \
    --mlp-layer 2 --use-norm \
    --weight-decay 5e-4 \
    --warmup-stage -1 > sign_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 2 --aggr-gpu -1 \
    --model sign --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --input-drop 0.0 --use-norm \
    --K 2 --weight-decay 3e-6 \
    --warmup-stage -1 > sign_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 2 --aggr-gpu -1 \
    --model sign --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --input-drop 0.0 \
    --K 3 --weight-decay 0 \
    --warmup-stage -1 > sign_reddit.out &

    #ppi : cannot start
    nohup python -u sagn.py \
    --gpu 1 --aggr-gpu -1 \
    --model sign --seed 0 \
    --dataset ppi --inductive \
    --threshold 0.95 0.95 --zero-inits \
    --epoch-setting 2000--eval-every 20 \
    --lr 0.001 --batch-size 256 \
    --mlp-layer 3 --num-hidden 1024 \
    --dropout 0.3 --input-drop 0.0 \
    --K 2 --weight-decay 3e-6 \
    --warmup-stage -1  > sign_ppi.out &

    #yelp : not yet
    nohup python -u sagn.py \
    --seed 0 --num-runs 10 \
    --model sign \
    --aggr-gpu -1 --gpu 0 \
    --dataset yelp --inductive \
    --zero-inits --eval-every 1 \
    --threshold 0.9 --epoch-setting 100 \
    --lr 0.0001 --batch-size 20000 \
    --eval-batch-size 200000 --mlp-layer 2 \
    --num-hidden 512 --dropout 0.1 \
    --input-drop 0.0 --K 2 \
    --weight-decay 5e-6 \
    --warmup-stage -1  > sign_yelp.out &


# hce_sign : true/false (Maybe false is better)
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 1 --aggr-gpu -1 \
    --seed 0 --model HCE_sign_concate_false \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 1200 \
    --eval-batch-size 1200 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 1e-6 \
    --warmup-stage -1 > HCE_sign_false_products.out &

    #cora
    nohup python -u sagn.py \
    --dataset cora \
    --gpu 1 --aggr-gpu -1 \
    --model HCE_sign_concate_false \
    --acc-loss acc --threshold 0.9 \
    --eval-every 10 --num-runs 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 50 --num-hidden 64 \
    --dropout 0.5 --input-drop 0.\
    --K 9 --mlp-layer 2 \
    --use-norm --weight-decay 5e-4 \
    --warmup-stage -1 >  HCE_sign_false_cora.out &

    #flickr
    nohup python -u sagn.py \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sign_concate_true\
    --dataset flickr --zero-inits \
    --inductive --eval-every 10 \
    --num-runs 10 --epoch-setting 200 \
    --threshold 0.5 --lr 0.001 \
    --batch-size 256 --eval-batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --input-drop 0.0 --use-norm \
    --K 2 --weight-decay 3e-6 \
    --warmup-stage -1 > HCE_sign_true_flickr.out &

    #reddit : not yet..
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sign_concate_false \
    --inductive --zero-inits \
    --threshold 0.9 0.9  --eval-every 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 10000 --eval-batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --input-drop 0.0 \
    --K 3 --weight-decay 0 \
    --warmup-stage -1 > HCE_sign_false_reddit.out &



#expert_sign
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 1 --aggr-gpu -1 \
    --seed 0 --model Expert_sign \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 20000 \
    --eval-batch-size 20000 \
    --num-hidden 512 --dropout 0.4 \
    --input-drop 0.3 --K 5 \
    --weight-decay 1e-6 \
    --warmup-stage -1 > Expert_sign_products.out &