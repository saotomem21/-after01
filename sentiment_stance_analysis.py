from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import pandas as pd

class SentimentStanceAnalyzer:
    def __init__(self, model_path="./domain_adapted_model"):
        # Load domain-adapted model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        )
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=5
        )
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model.to(self.device)
        self.stance_model.to(self.device)
        
        # Sentiment labels
        self.sentiment_labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        
        # Stance labels (Japanese)
        self.stance_labels = [
            "強く賛成", 
            "賛成", 
            "中立", 
            "反対", 
            "強く反対"
        ]

    def _calculate_dynamic_threshold(self, text_length):
        """Calculate dynamic threshold based on text length"""
        length_factor = max(0.5, min(1.0, text_length / 20))
        return 0.5 + (0.03 * length_factor)

    def _normalize_score(self, score, threshold):
        """Normalize score based on threshold"""
        if score > threshold:
            return (score - threshold) / (1.0 - threshold)
        return (score - 0.0) / threshold

    def analyze_sentiment(self, content):
        """Analyze sentiment with dynamic threshold and score normalization"""
        inputs = self.tokenizer(
            content, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            max_prob = torch.max(probs).item()
            
        # Calculate dynamic threshold
        threshold = self._calculate_dynamic_threshold(len(content))
        
        # Normalize score
        normalized_score = self._normalize_score(max_prob, threshold)
        normalized_score = max(0.0, min(1.0, normalized_score))
        final_score = round(normalized_score, 3)
        
        # Get label
        label = self.sentiment_labels[pred]
        if max_prob < threshold:
            label = "neutral"
            
        return {
            "label": label,
            "score": final_score,
            "formatted": f"{label} ({final_score:.3f})"
        }

    def analyze_stance(self, content):
        """Analyze stance with softmax probabilities"""
        inputs = self.tokenizer(
            content, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.stance_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1)
            max_prob = torch.max(probs).item()
            
        # Round score
        final_score = round(max_prob, 3)
        
        return {
            "label": self.stance_labels[pred],
            "score": final_score,
            "formatted": f"{self.stance_labels[pred]} ({final_score:.3f})"
        }

    def analyze_batch_from_csv(self, input_csv, output_csv):
        """Analyze texts from an input CSV and save results to an output CSV"""
        try:
            df = pd.read_csv(input_csv)
            if '内容' not in df.columns:
                print(f"Error: 'text' column not found in {input_csv}.")
                return
            
            texts = df['内容'].tolist()  # Analyze all rows
            results = []
            print("Starting analysis...")
            for i, text in enumerate(texts):
                sentiment = self.analyze_sentiment(text)
                stance = self.analyze_stance(text)
                
                results.append({
                    "number": i + 1,
                    "content": text,
                    "sentiment": sentiment["formatted"],
                    "sentiment_score": sentiment["score"],
                    "stance": stance["formatted"],
                    "stance_score": stance["score"]
                })
                
            result_df = pd.DataFrame(results)
            result_df.to_csv(output_csv, index=False)
            print(f"Analysis results saved to {output_csv}")
        except Exception as e:
            print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    analyzer = SentimentStanceAnalyzer()
    input_path = "./csvファイル/combined_output.csv"  # Input CSV path
    output_path = "./csvファイル/analysis_results.csv"  # Output CSV path
    analyzer.analyze_batch_from_csv(input_path, output_path)
