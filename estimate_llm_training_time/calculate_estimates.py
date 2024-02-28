#https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/


import pandas as pd

df = pd.read_csv('estimate_llm_training_time\\input.tsv', delimiter='\t')

# Define a list of GPU counts
gpu_counts = [1024]
#gpu_counts = [1]

# Create an empty DataFrame to store the results
result_df = pd.DataFrame()
BILLION = 1e9
# Iterate through each GPU count and perform calculations
for gpu_count in gpu_counts:
    temp_df = df.copy()  # Make a copy of the original DataFrame
    
    # Update GPU count column
    temp_df['GPU Count'] = gpu_count
    
    # Convert 'Estimated teraFLOPS' to FLOPS
    flops_per_gpu = temp_df['teraFlOPs per GPU'] * 1e12

    model_parmameters = temp_df['Number of Parameters (Billions)'] * BILLION
    
    tokens = temp_df['Number of Tokens (Billions)'] * BILLION

    # Calculate 'Training Time (Days)' for the current GPU count
    training_time_seconds = 8 * ((model_parmameters*tokens)/(gpu_count * flops_per_gpu))
    
    temp_df['Training Time (Days)'] = training_time_seconds/86400 #86400 = number of seconds in the day

    # # Calculate 'Data Transfer Rate (GBits/second)'
    temp_df['Data Transfer Rate (GBits/second)'] = (temp_df['Dataset Size (GB)'] * temp_df['Number of Epochs'] * 8) / training_time_seconds 
 
    
    # Append the results for the current GPU count to the overall results DataFrame
    result_df = result_df._append(temp_df)

# Print the overall result DataFrame
print(result_df)

# Save to a TSV file
result_df.to_csv('estimate_llm_training_time\\output.tsv', sep='\t', index=False)
