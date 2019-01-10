#!/bin/bash
_ratio=(0.01 0.02 0.03 0.05 0.1 0.2 0.5 0.7 0.9)
_lr=(1e-5 3e-5 5e-5 7e-5 1e-4)
_nb=(50 100)
_epoch=(50 100 200)
_loss=(BCELoss BCEWithLogitsLoss MSELoss)
testTime=10

for ratio in ${_ratio[@]}; do
    for lr in ${_lr[@]}; do
        for nb in ${_nb[@]}; do
            for e in ${_epoch[@]}; do
                for (( i = 0; i < $testTime; i++ )); do
                    python ${1} ${2} -r ${ratio} -nb ${nb} -e ${e} -lr ${lr} -noP
                    echo ${i}
                done
            done
        done
    done
done
