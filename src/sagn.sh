 #sagn without SLE
    #ogbn_products
nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --model sagn --zero-inits \
    --seed 0 --num-runs 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 50000 --num-hidden 512 \
    --dropout 0.5 --attn-drop 0.4 \
    --input-drop 0.2 --K 5 \
    --weight-decay 0 --warmup-stage -1 > sagn_products.out &

    #cora
nohup python -u sagn.py \
    --dataset cora \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50 \
    --num-hidden 64 --dropout 0.5 \
    --attn-drop 0. --input-drop 0.\
    --K 9 --mlp-layer 2 \
    --use-norm --weight-decay 5e-4 \
    --warmup-stage -1 > sagn_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --attn-drop 0.0 --input-drop 0.0 \
    --use-norm --K 2 \
    --weight-decay 3e-6 \
    --warmup-stage -1 > sagn_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --attn-drop 0.4 \
    --input-drop 0.0 --K 3 \
    --weight-decay 0 \
    --warmup-stage -1 > sagn_reddit.out &

    #ppi : cannot start
    nohup python -u sagn.py \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --seed 0 \
    --dataset ppi --inductive \
    --threshold 0.95 0.95 --zero-inits \
    --epoch-setting 2000 --eval-every 20 \
    --lr 0.001 --batch-size 256 \
    --mlp-layer 3 --num-hidden 1024 \
    --dropout 0.3 --attn-drop 0.1 \
    --input-drop 0.0 --K 2 \
    --weight-decay 3e-6 \
    --warmup-stage -1  > sagn_ppi.out &

    #yelp : not yet
    nohup python -u sagn.py \
    --seed 0 --num-runs 10 \
    --model sagn \
    --aggr-gpu -1 --gpu 0 \
    --dataset yelp --inductive \
    --zero-inits --eval-every 1 \
    --threshold 0.9 --epoch-setting 100 \
    --lr 0.0001 --batch-size 20000 \
    --eval-batch-size 200000 --mlp-layer 2 \
    --num-hidden 512 --dropout 0.1 \
    --attn-drop 0.0 --input-drop 0.0 \
    --K 2 --weight-decay 5e-6 \
    --warmup-stage -1  > sagn_yelp.out &

# hce_sagn : true/false (Maybe false is better) : What happen to you?
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sagn_false --zero-inits \
    --seed 0 --num-runs 5 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 1200 --eval-batch-size 1200 \
    --num-hidden 512 \
    --dropout 0.5 --attn-drop 0.4 \
    --input-drop 0.2 --K 5 \
    --weight-decay 0 --warmup-stage -1 > HCE_sagn_false_products.out &

    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 1 --aggr-gpu -1 \
    --model HCE_sagn_true --zero-inits \
    --seed 0 --num-runs 5 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 1200 --eval-batch-size 1200 \
    --num-hidden 512 \
    --dropout 0.5 --attn-drop 0.4 \
    --input-drop 0.2 --K 5 \
    --weight-decay 0 --warmup-stage -1 > HCE_sagn_true_products.out &

    #cora
nohup python -u sagn.py \
    --dataset cora \
    --gpu 1 --aggr-gpu -1 \
    --model HCE_sagn_false --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 50 \
    --num-hidden 64 --dropout 0.5 \
    --attn-drop 0. --input-drop 0.\
    --K 9 --mlp-layer 2 \
    --use-norm --weight-decay 5e-4 \
    --warmup-stage -1 > HCE_sagn_false_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --attn-drop 0.0 --input-drop 0.0 \
    --use-norm --K 2 \
    --weight-decay 3e-6 \
    --warmup-stage -1 > sagn_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 1 --aggr-gpu -1 \
    --model sagn --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --attn-drop 0.4 \
    --input-drop 0.0 --K 3 \
    --weight-decay 0 \
    --warmup-stage -1 > sagn_reddit.out &
