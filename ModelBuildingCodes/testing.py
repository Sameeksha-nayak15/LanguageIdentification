"""
Comprehensive Testing Script for Language Identifier (Joblib Version)
Tests multilingual texts, code-mixed inputs, and generates evaluation metrics
Works with pipeline .joblib file
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    f1_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import joblib
import json
from collections import Counter

class LanguageIdentifierTester:
    def __init__(self, pipeline_path='saved_models/language_pipeline.joblib'):
        """
        Initialize tester with trained pipeline from joblib
        
        Args:
            pipeline_path: Path to .joblib file containing the complete pipeline
        """
        print(f"Loading pipeline from {pipeline_path}...")
        self.pipeline = joblib.load(pipeline_path)
        print(f"✓ Pipeline loaded successfully")
        
        # Extract components from pipeline
        if hasattr(self.pipeline, 'named_steps'):
            # It's a sklearn Pipeline
            print("✓ Detected sklearn Pipeline")
            self.vectorizer = self.pipeline.named_steps.get('vectorizer') or self.pipeline.named_steps.get('tfidf')
            self.model = self.pipeline.named_steps.get('classifier') or self.pipeline.named_steps.get('model')
            
            if self.vectorizer:
                print(f"✓ Extracted vectorizer: {type(self.vectorizer).__name__}")
            if self.model:
                print(f"✓ Extracted model: {type(self.model).__name__}")
        else:
            # Direct pipeline object
            self.vectorizer = None
            self.model = None
            print("✓ Using pipeline directly for predictions")
        
        # Get language classes
        self.languages = None
        if self.model and hasattr(self.model, 'classes_'):
            self.languages = self.model.classes_
        elif hasattr(self.pipeline, 'classes_'):
            self.languages = self.pipeline.classes_
        
        if self.languages is not None:
            print(f"✓ Found {len(self.languages)} languages: {', '.join(map(str, self.languages[:10]))}{'...' if len(self.languages) > 10 else ''}")
        else:
            print("⚠ Warning: Could not determine language classes from pipeline")
        
        self.results = {}
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict languages for a list of texts"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Use pipeline directly for prediction
            predictions = self.pipeline.predict(texts)
            return list(predictions)
            
        except Exception as e:
            print(f"\nError during prediction: {e}")
            print(f"Pipeline type: {type(self.pipeline)}")
            raise
    
    def predict_with_confidence(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict languages with confidence scores"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Predict using pipeline
            predictions = self.pipeline.predict(texts)
            
            # Get probabilities/confidence scores
            if hasattr(self.pipeline, 'predict_proba'):
                # Pipeline has predict_proba (e.g., with probability-based classifiers)
                probabilities = self.pipeline.predict_proba(texts)
            elif hasattr(self.pipeline, 'decision_function'):
                # Pipeline has decision_function (e.g., SVM)
                decision = self.pipeline.decision_function(texts)
                # Convert decision scores to confidence scores
                if len(decision.shape) == 1:
                    # Binary classification
                    probabilities = np.vstack([-decision, decision]).T
                else:
                    # Multi-class
                    probabilities = decision
                # Normalize to [0, 1] range (relative confidence)
                prob_min = probabilities.min(axis=1, keepdims=True)
                prob_max = probabilities.max(axis=1, keepdims=True)
                probabilities = (probabilities - prob_min) / (prob_max - prob_min + 1e-10)
            else:
                # No confidence scores available
                probabilities = np.ones((len(texts), 1))
            
            # Extract confidence scores
            results = []
            for i, pred in enumerate(predictions):
                if probabilities.shape[1] > 1:
                    # Get max confidence
                    confidence = np.max(probabilities[i])
                else:
                    confidence = 1.0
                results.append((pred, confidence))
            
            return results
            
        except Exception as e:
            print(f"\nError during prediction with confidence: {e}")
            raise
    
    def test_multilingual_texts(self) -> Dict:
        """Test with various multilingual texts"""
        print("\n" + "="*80)
        print("MULTILINGUAL TEXT TESTS")
        print("="*80)
        
        # Comprehensive multilingual test set
        test_data = {
            'eng': [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is transforming the world.",
                "Python is a popular programming language."
            ],
            'spa': [
                "El rápido zorro marrón salta sobre el perro perezoso.",
                "El aprendizaje automático está transformando el mundo.",
                "Python es un lenguaje de programación popular."
            ],
            'fra': [
                "Le rapide renard brun saute par-dessus le chien paresseux.",
                "L'apprentissage automatique transforme le monde.",
                "Python est un langage de programmation populaire."
            ],
            'deu': [
                "Der schnelle braune Fuchs springt über den faulen Hund.",
                "Maschinelles Lernen verändert die Welt.",
                "Python ist eine beliebte Programmiersprache."
            ],
            'ita': [
                "La veloce volpe marrone salta sopra il cane pigro.",
                "L'apprendimento automatico sta trasformando il mondo.",
                "Python è un linguaggio di programmazione popolare."
            ],
            'por': [
                "A rápida raposa marrom pula sobre o cachorro preguiçoso.",
                "O aprendizado de máquina está transformando o mundo.",
                "Python é uma linguagem de programação popular."
            ],
            'rus': [
                "Быстрая коричневая лиса прыгает через ленивую собаку.",
                "Машинное обучение меняет мир.",
                "Python - популярный язык программирования."
            ],
            'zho': [
                "敏捷的棕色狐狸跳过懒狗。",
                "机器学习正在改变世界。",
                "Python是一种流行的编程语言。"
            ],
            'jpn': [
                "素早い茶色のキツネが怠け者の犬を飛び越える。",
                "機械学習は世界を変えています。",
                "Pythonは人気のあるプログラミング言語です。"
            ],
            'ara': [
                "الثعلب البني السريع يقفز فوق الكلب الكسول.",
                "التعلم الآلي يغير العالم.",
                "بايثون هي لغة برمجة شعبية."
            ],
            'hin': [
                "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
                "मशीन लर्निंग दुनिया को बदल रहा है।",
                "पायथन एक लोकप्रिय प्रोग्रामिंग भाषा है।"
            ],
            'kor': [
                "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
                "머신 러닝이 세상을 변화시키고 있습니다.",
                "파이썬은 인기있는 프로그래밍 언어입니다."
            ]
        }
        
        all_texts = []
        all_labels = []
        
        for lang, texts in test_data.items():
            all_texts.extend(texts)
            all_labels.extend([lang] * len(texts))
        
        # Predict
        predictions = self.predict(all_texts)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'text': all_texts,
            'true_label': all_labels,
            'predicted_label': predictions,
            'correct': [t == p for t, p in zip(all_labels, predictions)]
        })
        
        print(f"\nAccuracy: {accuracy*100:.2f}%")
        print(f"Correct: {results_df['correct'].sum()}/{len(results_df)}")
        
        # Language-wise accuracy
        print("\nPer-Language Results:")
        for lang in test_data.keys():
            lang_df = results_df[results_df['true_label'] == lang]
            if len(lang_df) > 0:
                lang_acc = lang_df['correct'].mean() * 100
                print(f"  {lang}: {lang_acc:.2f}% ({lang_df['correct'].sum()}/{len(lang_df)})")
        
        self.results['multilingual'] = {
            'accuracy': accuracy,
            'results_df': results_df,
            'test_data': test_data
        }
        
        return results_df
    
    def test_code_mixed_inputs(self) -> pd.DataFrame:
        """Test with code-mixed/bilingual texts"""
        print("\n" + "="*80)
        print("CODE-MIXED INPUT TESTS")
        print("="*80)
        
        code_mixed_texts = [
            # English-Spanish
            ("I love programming, pero también me gusta el español.", "eng_spa_mixed"),
            ("Let's go to the playa tomorrow.", "eng_spa_mixed"),
            
            # English-French
            ("Hello, comment allez-vous today?", "eng_fra_mixed"),
            ("C'est very interesting, n'est-ce pas?", "eng_fra_mixed"),
            
            # English-German
            ("Ich bin learning Python programming.", "eng_deu_mixed"),
            ("Das ist really cool!", "eng_deu_mixed"),
            
            # English-Japanese
            ("Hello world, こんにちは世界", "eng_jpn_mixed"),
            ("I love sushi とても美味しい", "eng_jpn_mixed"),
            
            # English-Hindi
            ("मैं Python सीख रहा हूं and enjoying it.", "eng_hin_mixed"),
            ("This is बहुत अच्छा!", "eng_hin_mixed"),
            
            # Spanish-French
            ("Hola, comment ça va?", "spa_fra_mixed"),
            
            # Multi-language
            ("Hello नमस्ते 你好 Bonjour", "multi_mixed"),
        ]
        
        texts = [t[0] for t in code_mixed_texts]
        expected_types = [t[1] for t in code_mixed_texts]
        
        # Predict with confidence
        predictions = self.predict_with_confidence(texts)
        
        results_df = pd.DataFrame({
            'text': texts,
            'expected_type': expected_types,
            'predicted_language': [p[0] for p in predictions],
            'confidence': [p[1] for p in predictions]
        })
        
        print("\nCode-Mixed Text Results:")
        print(results_df.to_string(index=False))
        
        avg_confidence = results_df['confidence'].mean()
        print(f"\nAverage Confidence: {avg_confidence*100:.2f}%")
        print("\nNote: Lower confidence is expected for code-mixed texts.")
        
        self.results['code_mixed'] = results_df
        
        return results_df
    
    def test_edge_cases(self) -> pd.DataFrame:
        """Test edge cases"""
        print("\n" + "="*80)
        print("EDGE CASE TESTS")
        print("="*80)
        
        edge_cases = [
            ("Hi", "very_short"),
            ("a", "single_char"),
            ("123456789", "numbers_only"),
            ("!@#$%^&*()", "punctuation_only"),
            ("Hello мир 世界", "multi_script"),
            ("https://www.example.com", "url"),
            ("test@example.com", "email"),
            ("😀😁😂🤣", "emoji_only"),
            ("       ", "whitespace_only"),
            ("The " * 100, "repetitive"),
        ]
        
        texts = [e[0] for e in edge_cases]
        case_types = [e[1] for e in edge_cases]
        
        predictions = self.predict_with_confidence(texts)
        
        results_df = pd.DataFrame({
            'case_type': case_types,
            'text': [t[:50] + ('...' if len(t) > 50 else '') for t in texts],
            'predicted_language': [p[0] for p in predictions],
            'confidence': [p[1] for p in predictions]
        })
        
        print("\nEdge Case Results:")
        print(results_df.to_string(index=False))
        
        self.results['edge_cases'] = results_df
        
        return results_df
    
    def evaluate_on_test_set(self, X_test, y_test) -> Dict:
        """Complete evaluation on test set"""
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)
        
        # Predict
        predictions = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:            {accuracy*100:.2f}%")
        print(f"  F1-Score (Macro):    {f1_macro:.4f}")
        print(f"  F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"  Precision:           {precision:.4f}")
        print(f"  Recall:              {recall:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Classification report
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        
        # Detailed results
        results_df = pd.DataFrame({
            'text': X_test,
            'true_label': y_test,
            'predicted_label': predictions,
            'correct': [t == p for t, p in zip(y_test, predictions)]
        })
        
        # Separate correct and incorrect
        correct_df = results_df[results_df['correct'] == True]
        incorrect_df = results_df[results_df['correct'] == False]
        
        print(f"\nCorrect Predictions: {len(correct_df)}")
        print(f"Incorrect Predictions: {len(incorrect_df)}")
        
        self.results['test_set'] = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': report,
            'results_df': results_df,
            'correct_df': correct_df,
            'incorrect_df': incorrect_df
        }
        
        return self.results['test_set']
    
    def create_confusion_matrix_plot(self, y_test, predictions, save_path='confusion_matrix.png'):
        """Create and save confusion matrix visualization"""
        # Get unique labels
        labels = sorted(list(set(y_test) | set(predictions)))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=labels)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Language', fontsize=12)
        plt.ylabel('True Language', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {save_path}")
        plt.close()
    
    def create_results_table(self, results_df: pd.DataFrame, save_path='results_table.csv'):
        """Create table of correctly identified vs misclassified languages"""
        print("\n" + "="*80)
        print("RESULTS TABLE: CORRECT vs MISCLASSIFIED")
        print("="*80)
        
        # Summary by language
        summary = []
        for lang in sorted(results_df['true_label'].unique()):
            lang_df = results_df[results_df['true_label'] == lang]
            correct = lang_df['correct'].sum()
            total = len(lang_df)
            incorrect = total - correct
            accuracy = (correct / total * 100) if total > 0 else 0
            
            summary.append({
                'Language': lang,
                'Total_Samples': total,
                'Correctly_Identified': correct,
                'Misclassified': incorrect,
                'Accuracy_%': accuracy
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('Accuracy_%', ascending=False)
        
        print("\nPer-Language Summary:")
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv(save_path, index=False)
        print(f"\nResults table saved to {save_path}")
        
        # Also save detailed results
        detailed_path = save_path.replace('.csv', '_detailed.csv')
        results_df.to_csv(detailed_path, index=False)
        print(f"Detailed results saved to {detailed_path}")
        
        return summary_df
    
    def generate_full_report(self, save_dir='evaluation_results'):
        """Generate comprehensive evaluation report"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Save all results
        report = {
            'summary': {
                'multilingual_accuracy': self.results.get('multilingual', {}).get('accuracy'),
                'test_set_accuracy': self.results.get('test_set', {}).get('accuracy'),
                'total_tests_run': len(self.results)
            }
        }
        
        # Save as JSON
        with open(f'{save_dir}/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nAll results saved to {save_dir}/")
        print("\nFiles generated:")
        print(f"  - evaluation_report.json")
        
        if 'multilingual' in self.results:
            print(f"  - Multilingual test results available")
        if 'code_mixed' in self.results:
            self.results['code_mixed'].to_csv(f'{save_dir}/code_mixed_results.csv', index=False)
            print(f"  - code_mixed_results.csv")
        if 'edge_cases' in self.results:
            self.results['edge_cases'].to_csv(f'{save_dir}/edge_case_results.csv', index=False)
            print(f"  - edge_case_results.csv")
        
        return report


def main():
    """Main testing pipeline"""
    print("="*80)
    print("LANGUAGE IDENTIFIER - COMPREHENSIVE TESTING SUITE (PIPELINE)")
    print("="*80)
    
    # Initialize tester with your pipeline .joblib file
    pipeline_path = 'saved_models/language_pipeline.joblib'  # Update this path
    
    tester = LanguageIdentifierTester(pipeline_path=pipeline_path)
    
    # Run all tests
    print("\n[1/4] Running Multilingual Tests...")
    multilingual_results = tester.test_multilingual_texts()
    
    print("\n[2/4] Running Code-Mixed Tests...")
    code_mixed_results = tester.test_code_mixed_inputs()
    
    print("\n[3/4] Running Edge Case Tests...")
    edge_case_results = tester.test_edge_cases()
    
    # If you have a test set, load and evaluate
    try:
        print("\n[4/4] Loading Test Set...")
        # Example: Load from your test data
        # test_df = pd.read_csv('test_data.csv')
        # X_test = test_df['text'].tolist()
        # y_test = test_df['language'].tolist()
        
        # test_results = tester.evaluate_on_test_set(X_test, y_test)
        # tester.create_confusion_matrix_plot(y_test, tester.predict(X_test))
        # tester.create_results_table(test_results['results_df'])
        
        print("\nTest set evaluation skipped (no test data loaded)")
        print("To use test set evaluation, uncomment and modify the code above")
    except Exception as e:
        print(f"\nTest set evaluation skipped: {e}")
    
    # Generate full report
    print("\n[5/5] Generating Comprehensive Report...")
    tester.generate_full_report()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print("\nTo test with your own text:")
    print(">>> text = 'Your text here'")
    print(">>> prediction = tester.predict([text])")
    print(">>> print(prediction)")


if __name__ == "__main__":
    main()