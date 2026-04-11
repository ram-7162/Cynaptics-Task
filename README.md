-- model.py # Transformer architecture
-- train.py # Training loop
-- generate.py # Text generation script
-- input.txt # Dataset (Tiny Shakespeare)


## Features

- Custom implementation of:
  - Masked Self-Attention
  - Multi-Head Attention
  - Feed Forward Networks
  - Layer Normalization
- GPT-2 style architecture
- Subword tokenization using GPT-2 tokenizer



##  Tokenization

Initial Approach:
- Character-level tokenization

Updated Approach:
- Switched to **GPT-2 subword tokenizer**


### Data Scaling
  text = text * 5


## Training Results
<img width="1356" height="1014" alt="Screenshot 2026-04-10 223843" src="https://github.com/user-attachments/assets/90f15363-4e09-448a-a85f-0dec2795f38e" />
<img width="1254" height="1010" alt="Screenshot 2026-04-10 231040" src="https://github.com/user-attachments/assets/bfb95648-2c33-444d-8e6f-d9bf37beae69" />



## generated text

<img width="1238" height="982" alt="Screenshot 2026-04-10 231454" src="https://github.com/user-attachments/assets/5833aa3c-7a8c-4659-b862-cacd3bb61e39" />

