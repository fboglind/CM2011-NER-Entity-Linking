from datasets import load_dataset

ds = load_dataset("community-datasets/swedish_medical_ner", "1177")

print(ds)  # shows splits and number of examples
print(ds["train"].features)  # schema
print(ds["train"][0])        # first example
print(ds["train"][1])  
print(ds["train"][2]) 
