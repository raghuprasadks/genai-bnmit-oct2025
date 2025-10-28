
# pip install transformers torch torchvision torchaudio
# pip install tf-keras
from transformers import pipeline

# Load pre-trained summarization pipeline
summarization_pipeline = pipeline("summarization")

# Function to summarize text
def summarize_text(text, max_length=100, min_length=30):
    summary = summarization_pipeline(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example passage for summarization
long_text = (
    "Artificial intelligence is rapidly transforming various industries, "
    "enabling automation, improving efficiency, and enhancing decision-making processes. "
    "With the rise of machine learning and deep learning models, businesses can now process large amounts of data "
    "more effectively, uncover hidden patterns, and gain valuable insights. "
    "However, ethical concerns, data privacy, and bias in AI algorithms remain significant challenges. "
    "As technology advances, organizations and policymakers must collaborate to ensure responsible AI development and deployment."
)

# Obtain summarized text
summary_result = summarize_text(long_text)
print("Original Text:", long_text)
print("Summarized Text:", summary_result)
