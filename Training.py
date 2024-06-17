import torch
from torch.utils.data import DataLoader

# Preper data loder

def collate_fn(batch):
  input_ids = [processor(text=item["text"],return_tensors="pt").input_ids for item in batch]
  labels = [item["mel_spectrogram"] for item in batch]
  input_ids = torch.cat(input_ids)
  labels = torch.stack(labels)
  return {"input_ids": input_ids, "labels": labels}


dataloader = DataLoader(dataset["train"], batch_size=8, collate_fn=collate_fn)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(10):  # Train for 10 epochs
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



  

