#xnor_case1_sagn
    #ogbn_products
    nohup python -u sagn.py --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --model xnor_case1_sagn --zero-inits \
    --seed 0 --num-runs 10 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 50000 --num-hidden 512 \
    --dropout 0.5 --attn-drop 0.4 \
    --input-drop 0.2 --K 5 \
    --weight-decay 0 --warmup-stage -1 > xnor_case1_sagn_whole.out &

    #cora : ?? too much low acc
nohup python -u sagn.py \
    --dataset cora \
    --gpu 0 --aggr-gpu -1 \
    --model xnor_case1_sagn --acc-loss acc \
    --threshold 0.9 --eval-every 10 \
    --num-runs 10 --epoch-setting 1000 \
    --lr 0.01 --batch-size 140 \
    --num-hidden 64 --dropout 0.5 \
    --attn-drop 0. --input-drop 0.\
    --K 9 --mlp-layer 2 \
    --use-norm --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sagn_cora.out &

    #flickr
nohup python -u sagn.py \
    --gpu 0 --aggr-gpu -1 \
    --model xnor_case1_sagn --dataset flickr \
    --zero-inits --inductive \
    --eval-every 8 --num-runs 10 \
    --epoch-setting 200 --threshold 0.5 \
    --lr 0.001 --batch-size 256 \
    --num-hidden 512 --dropout 0.7 \
    --attn-drop 0.0 --input-drop 0.0 \
    --use-norm --K 2 \
    --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sagn_flickr.out &

    #reddit
nohup python -u sagn.py \
    --dataset reddit --num-runs 10 \
    --gpu 1 --aggr-gpu -1 \
    --model xnor_case1_sagn --inductive \
    --zero-inits --threshold 0.9 0.9 \
    --eval-every 10 --epoch-setting 1000 \
    --lr 0.001 --batch-size 10000 \
    --num-hidden 512 --num-heads 1 \
    --dropout 0.7 --attn-drop 0.4 \
    --input-drop 0.0 --K 3 \
    --weight-decay 0 \
    --warmup-stage -1 > xnor_case1_sagn_reddit.out &

#hce_sagn_xnor_case1 (false)
    #ogbn_products
    nohup python -u sagn.py \
    --dataset ogbn-products \
    --gpu 0 --aggr-gpu -1 \
    --model HCE_sagn_xnor_case1 --zero-inits \
    --seed 0 --num-runs 5 \
    --epoch-setting 1000 --lr 0.001 \
    --batch-size 600 --eval-batch-size 600 \
    --num-hidden 512 \
    --dropout 0.5 --attn-drop 0.4 \
    --input-drop 0.2 --K 5 \
    --weight-decay 0 --warmup-stage -1 > HCE_sagn_xnor_case1_products.out &