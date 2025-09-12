#!/bin/bash

# # Array of classifier types
# classifiers=("arcface" "cosface" "FC")

# # Array of optimizers
# # optimizers=("adamw" "SGD")
# optimizers=("adamw")

# # Loop through each combination
# for classifier in "${classifiers[@]}"; do
#     for optimizer in "${optimizers[@]}"; do
#         echo "Running with classifier: $classifier, optimizer: $optimizer"
#         python fine_tune_main.py \
#             --model_name ir_se \
#             --save_plot \
#             --classifier_type "$classifier" \
#             --phase last_block \
#             --optimizer "$optimizer"
#     done
# done

python fine_tune_main.py --model_name mobilefacenet --save_plot --classifier_type "arcface" --phase last_block --optimizer adamw
