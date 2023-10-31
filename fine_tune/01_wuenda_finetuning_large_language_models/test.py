from llama import BasicModelRunner

non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")

non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")

print(non_finetuned_output)


