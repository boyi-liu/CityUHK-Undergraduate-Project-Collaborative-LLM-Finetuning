# FedLLM Factory

## 1. Generate the dataset

Configure the dataset split in `dataset/config.yaml`, then run the matching
generator:

```bash
cd dataset
python3 generate_<dataset>.py
cd ..
```

## 2. Configure the experiment

Edit `config.yaml`. Command-line arguments override values from that file.

- Algorithms: `fedit`, `fedasync`, `ffalora`
- Standard mode: `prototype`
- Event-driven mode: `realistic`

## 3. Run training

Standard version:

```bash
python3 main.py --alg fedit --mode prototype --epoch 1
```

Event-driven version:

```bash
python3 main.py --alg fedit --mode realistic --epoch 1
```

To run the experiments configured in `run.sh`:

```bash
bash run.sh
```

## 4. Evaluate the saved adapter

Use the same experiment arguments that were used during training:

```bash
python3 eval.py --alg fedit --dataset sst2 --model roberta-base --epoch 1
```
