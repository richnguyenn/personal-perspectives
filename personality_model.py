"""Big Five personality prediction using transformer models.

This module loads a Hugging Face model and predicts the Big Five personality
traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
from text. Supports pre-trained Big Five models and base models like roberta-base
with a custom regression head.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

# The five Big Five personality traits in standard order
BIG_FIVE_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


class PersonalityPredictor:
    """Predicts Big Five personality traits from text using a transformer model."""

    def __init__(self, model_name, device=None):
        """Initialize the predictor with a model.

        Parameters
        ----------
        model_name : str
            Hugging Face model ID (e.g., "vladinc/bigfive-regression-model"
            or "roberta-base").
        device : str or torch.device, optional
            Device to run the model on. If None, uses CUDA if available else CPU.
        """
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        """Load the tokenizer and model from Hugging Face."""
        print("Loading tokenizer from:", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Try to load as a sequence classification model first (for pre-trained Big Five models)
        self.use_custom_head = True
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if self.model.config.num_labels == 5:
                self.use_custom_head = False
                self.regression_head = None
            else:
                self.model = AutoModel.from_pretrained(self.model_name)
                self._add_regression_head()
        except Exception:
            self.model = AutoModel.from_pretrained(self.model_name)
            self._add_regression_head()

        self.model.to(self.device)
        if self.use_custom_head:
            self.regression_head.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def _add_regression_head(self):
        """Add a regression head for base models that don't have one."""
        # Get the hidden size from the model
        if hasattr(self.model, "config"):
            hidden_size = self.model.config.hidden_size
        else:
            hidden_size = 768

        # Create a simple linear layer: hidden_size -> 5
        self.regression_head = torch.nn.Linear(hidden_size, 5)
        self.regression_head.to(self.device)
        self.regression_head.eval()

    def predict(self, text):
        """Predict Big Five scores for a single text.

        Parameters
        ----------
        text : str
            The text to analyze (e.g., a Reddit comment).

        Returns
        -------
        dict
            Dictionary with keys for each trait. Values are floats between 0 and 1.
        """
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts):
        """Predict Big Five scores for a list of texts.

        Parameters
        ----------
        texts : list of str
            List of texts to analyze.

        Returns
        -------
        list of dict
            List of dictionaries. Each dict has keys for each trait.
        """
        # Skip empty texts
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and str(text).strip():
                valid_texts.append(str(text).strip())
                valid_indices.append(i)

        if not valid_texts:
            # Return zeros for all if no valid text
            empty_result = {trait: 0.0 for trait in BIG_FIVE_TRAITS}
            return [empty_result for _ in texts]

        # Tokenize
        inputs = self.tokenizer(
            valid_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.use_custom_head:
                outputs = self.model(**inputs)
                # Use [CLS] token or mean pooling
                if hasattr(outputs, "last_hidden_state"):
                    pooled = outputs.last_hidden_state[:, 0, :]
                else:
                    pooled = outputs.pooler_output
                logits = self.regression_head(pooled)
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Apply sigmoid to get values between 0 and 1
            scores = torch.sigmoid(logits)

        # Build results for valid texts
        valid_results = []
        for i in range(len(valid_texts)):
            row = {}
            for j, trait in enumerate(BIG_FIVE_TRAITS):
                row[trait] = float(scores[i][j].cpu())
            valid_results.append(row)

        # Map back to original order (fill in empty results for skipped texts)
        all_results = []
        result_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                all_results.append(valid_results[result_idx])
                result_idx += 1
            else:
                all_results.append({trait: 0.0 for trait in BIG_FIVE_TRAITS})

        return all_results
