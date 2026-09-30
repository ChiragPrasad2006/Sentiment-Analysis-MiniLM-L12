# Sentiment-Analysis-MiniLM-L12
A full trained model of MiniLM-L12 on Sentiment Analysis on 1.6 milion tweets with values ranging from 0-2 where 0=negetive &amp; 2=positive

url: https://huggingface.co/Kami0867/MiniLM-L12-Sentiment-Analysis/commit/faa356c2547697c7a2da8ce48ac50f904cd7da9e


# Load Model:

**model_id = "Kami0867/MiniLM-L12-Sentiment-Analysis"**

**tokenizer = AutoTokenizer.from_pretrained(model_id)**

**model = AutoModelForSequenceClassification.from_pretrained(model_id)**


# Dataset:

dataset: https://www.kaggle.com/datasets/kazanova/sentiment140


# Achieved accuracy:
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 2592.64it/s]

Map: 100%|████████████████████████████████████████████████████████████████████████████████████| 1440000/1440000 [02:08<00:00, 11214.66 examples/s]

Map: 100%|██████████████████████████████████████████████████████████████████████████████████████| 160000/160000 [00:14<00:00, 11246.56 examples/s]

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 22500/22500 [42:14<00:00,  8.88it/s]

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2500/2500 [04:41<00:00,  8.89it/s]

Train accuracy: 0.8881

Test accuracy:  0.8711

Train F1:       0.8881

Test F1:        0.8711