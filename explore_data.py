from datasets import load_dataset

ds = load_dataset("bigbio/swedish_medical_ner", "swedish_medical_ner_1177_bigbio_kb", trust_remote_code=True)

print(ds)  # shows splits and number of examples
print(ds["train"].features)  # schema
print(ds["train"][0])        # first example
print(ds["train"][1])  
print(ds["train"][2]) 
