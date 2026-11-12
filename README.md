# Corvid

<img src="logo.svg" alt="Corvid logo" width="64" height="64">

Lightweight, open-source specialized language model. Trained to write commit messages in software forges.

---

## System

Autoregressive transformer architecture implemented in TypeScript with TensorFlow.js. Designed to run in Node.js environments.

* Masked self-attention: each token attends only to previous tokens for sequential language modeling.
* Feed-forward network: dense layers applied after each attention block.
* Positional encoding: position information injection without recurrence or convolution.

---

## Model

Specialized intelligence for a specific purpose.

**Jackdaw XXS**

Trained to write commit messages in software forges.

| Property | Value |
| --- | --- |
| Parameters | 1.1M |
| Active parameters | 1.1M |
| Context | 256 |

---

## Training

Complete pipeline with data splitting, tokenization, loss function, and gradient optimization.

* BPE Tokenizer: byte pair encoding tokenization with serializable vocabulary.
* Cross entropy: cross-entropy loss over output distribution.
* Checkpoints: training save and resume from checkpoint.

---

## Technical Components

Independent modules for tokenizer, transformer, trainer, and model export.

* Tokenizer: encoding and decoding of text to token IDs.
* Transformer: self-attention, feed-forward, and layer normalization layers.
* Trainer: training loop with checkpoints and resume.
* Exporter: trained model export to ONNX format.

---

## API

Use the API like OpenAI. Change the base URL and API key to talk to Corvid.

List models:

```bash
curl https://tu-dominio.com/v1/models \
  -H "Authorization: Bearer tu-api-key"
```

Send a message:

```bash
curl https://tu-dominio.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{
    "model": "corvid-xxs",
    "messages": [
      {"role": "user", "content": "Write a commit to fix the login error"}
    ]
  }'
```

With the OpenAI client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://tu-dominio.com/v1",
    api_key="tu-api-key"
)

response = client.chat.completions.create(
    model="corvid-xxs",
    messages=[
        {"role": "user", "content": "Write a commit to fix the login error"}
    ]
)

print(response.choices[0].message.content)
```

---

## Installation

```bash
npm install
```

Train:

```bash
npx tsx scripts/train.ts
```

Test:

```bash
npm test
```

Type check:

```bash
npm run typecheck
```

---

## License

[MIT](LICENSE)
