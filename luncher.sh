model_names=(vits16 vits16_ft)
output_types=(Attention_Pooling cls)

for model_name in "${model_names[@]}"; do
    for output_type in "${output_types[@]}"; do
        if [ "$output_type" == "Attention_Pooling" ]; then
            echo "Skipping combination: model_name=$model_name and output_type=$output_type"
            continue
        fi
        echo "\n\nRunning DINO_SAC with model: $model_name and output: $output_type \n\n"
        python DINO_SAC.py extractor.model_name=$model_name extractor.output_type=$output_type
    done
done