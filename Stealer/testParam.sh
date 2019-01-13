#!/bin/bash
_ratio=(0.05 0.1 0.3 0.5 0.7 0.9)
_lr=(1e-5 5e-5 1e-4) #_lr=(1e-5 3e-5 5e-5 7e-5 1e-4)
_nb=(100) #_nb=(50 100)
_epoch=(200) # _epoch=(50 100 200)
_loss=(BCELoss) #_loss=(BCELoss BCEWithLogitsLoss MSELoss)
testTime=10
outfile="${2}_${1%.*}_${3}_result.csv"

title="training size,testing size,loss function,batch size,learning rate,epoch,control trainning accuracy,copy trainning accuracy,control testing accuracy,copy testing accuracy"

#echo -e "${1},${2},\c" | tee ${outfile}
#echo `date "+%Y-%m-%d %H:%M:%S"` | tee -a ${outfile} 
echo -e "${title}" | tee  ${outfile}

for ratio in ${_ratio[@]}; do
    for loss in ${_loss[@]}; do
        for nb in ${_nb[@]}; do
            for lr in ${_lr[@]}; do
                for e in ${_epoch[@]}; do
                    for (( i = 0; i < $testTime; i++ )); do
                        python ${1} ${2} -r ${ratio} -nb ${nb} -e ${e} -lr ${lr} -loss ${loss} -noP 2>&1 | tee -a ${outfile}
                    done
                done
            done
        done
    done
done

python getCsvAverage ${outfile}

#echo -e "${1},${2},\c" | tee  -a ${outfile}
#echo `date "+%Y-%m-%d %H:%M:%S"` | tee -a ${outfile}
