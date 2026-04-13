model_names=(vits16 vits16_ft)
output_types=(Attention_Pooling cls)
output_layer=(11 10 8 6)

for output_layer in "${output_layer[@]}"; do
    for model_name in "${model_names[@]}"; do
        for output_type in "${output_types[@]}"; do
            echo "\n\nRunning DINO_SAC with model: $model_name and output: $output_type layer: $output_layer\n\n"
            python DINO_SAC.py extractor.model_name=$model_name extractor.output_type=$output_type extractor.output_layer=$output_layer
        done
    done
done