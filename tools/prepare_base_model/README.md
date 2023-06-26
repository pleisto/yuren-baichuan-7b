# prepare-base-model

Prepare a base model for Yuren.

- Append Special Tokens to the end of the vocabulary.
- Resize the embedding size to be divisible by 128.
- Convert the baichuan model to a LLaMA model.

## How to use

Run the following command in the root directory of workspace.

```bash
rye run prepare-base-model
```
