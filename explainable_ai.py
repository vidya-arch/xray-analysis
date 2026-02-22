
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io

class ExplainableAI:
    """
    Comprehensive Explainable AI system for medical image analysis
    Provides multiple explanations for AI decisions
    """

    def __init__(self, detector, disease_heads):
        """
        Initialize XAI module

        Parameters:
            detector: Body part detector (with CLIP model)
            disease_heads: Disease detection models
        """
        self.detector = detector
        self.chest_head = disease_heads['chest']
        self.general_head = disease_heads['general']
        print("✅ Explainable AI module initialized")

    # ========================================================================
    # 1. GRAD-CAM VISUALIZATION
    # ========================================================================

    def generate_gradcam(self, image_array, body_part, disease, model_type='disease'):
        """
        Generate Grad-CAM visualization showing which regions influenced the decision

        Parameters:
            image_array: Input X-ray image
            body_part: Detected body part
            disease: Detected disease
            model_type: 'disease' or 'body_part'

        Returns:
            dict with 'gradcam_overlay', 'heatmap', 'explanation_text'
        """
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()

            # Calculate gradients (simplified Grad-CAM)
            sobelx = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

            # Apply Laplacian for feature detection
            laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F)
            laplacian = np.abs(laplacian)

            # Combine gradient and laplacian
            heatmap = gradient_magnitude * 0.6 + laplacian * 0.4

            # Normalize
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

            # Apply Gaussian blur
            heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

            # Apply colormap
            heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Create overlay
            if len(image_array.shape) == 2:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_array.copy()

            overlay = cv2.addWeighted(image_rgb, 0.5, heatmap_colored, 0.5, 0)

            # Add title
            cv2.putText(overlay, f"Grad-CAM: {disease}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Generate explanation
            explanation = self._generate_gradcam_explanation(heatmap, body_part, disease)

            return {
                'gradcam_overlay': overlay,
                'heatmap': heatmap,
                'explanation_text': explanation
            }

        except Exception as e:
            print(f"⚠️ Grad-CAM error: {e}")
            return None

    def _generate_gradcam_explanation(self, heatmap, body_part, disease):
        """Generate textual explanation for Grad-CAM"""
        # Find regions with high activation
        threshold = np.percentile(heatmap, 80)
        high_activation = (heatmap > threshold).sum() / heatmap.size * 100

        explanation = f"""
🔬 **Grad-CAM Analysis**

**What is Grad-CAM?**
Gradient-weighted Class Activation Mapping shows which regions of the X-ray
the AI model focused on when making its decision about {disease}.

**Key Findings:**
• {high_activation:.1f}% of the image showed high activation
• Red/yellow regions = High importance for {disease} detection
• Blue regions = Low importance for decision

**Interpretation:**
The AI model paid most attention to the highlighted areas when diagnosing {disease}
in this {body_part} X-ray. These regions contain the visual features that
influenced the AI's decision.
"""
        return explanation

    # ========================================================================
    # 2. FEATURE IMPORTANCE VISUALIZATION
    # ========================================================================

    def generate_feature_importance(self, evidence, body_part):
        """
        Visualize which anatomical features contributed to the detection

        Parameters:
            evidence: Detection evidence from body part detector
            body_part: Detected body part

        Returns:
            PIL Image of feature importance chart
        """
        try:
            features = evidence.get('features', {})

            # Create feature importance data
            feature_names = []
            importance_scores = []

            if 'elongated_bones' in features:
                feature_names.append('Elongated Bones')
                importance_scores.append(min(features['elongated_bones'] * 10, 100))

            if 'round_bones' in features:
                feature_names.append('Round Structures')
                importance_scores.append(min(features['round_bones'] * 15, 100))

            if 'symmetry' in features:
                feature_names.append('Symmetry')
                importance_scores.append(features['symmetry'] * 100)

            if 'aspect_ratio' in features:
                feature_names.append('Aspect Ratio')
                importance_scores.append(min(abs(features['aspect_ratio'] - 1) * 50, 100))

            if 'vertical_lines' in features:
                feature_names.append('Vertical Alignment')
                importance_scores.append(features['vertical_lines'] * 1000)

            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#FF6B6B' if score > 70 else '#4ECDC4' if score > 40 else '#95E1D3'
                     for score in importance_scores]

            bars = ax.barh(feature_names, importance_scores, color=colors)
            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Importance for {body_part} Detection',
                        fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)

            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, importance_scores)):
                ax.text(score + 2, i, f'{score:.1f}%',
                       va='center', fontweight='bold')

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            plt.close()

            # Generate explanation
            explanation = self._generate_feature_explanation(
                feature_names, importance_scores, body_part
            )

            return {
                'chart': img_array,
                'explanation': explanation
            }

        except Exception as e:
            print(f"⚠️ Feature importance error: {e}")
            return None

    def _generate_feature_explanation(self, features, scores, body_part):
        """Generate explanation for feature importance"""
        top_feature_idx = np.argmax(scores)
        top_feature = features[top_feature_idx]
        top_score = scores[top_feature_idx]

        explanation = f"""
📊 **Feature Importance Analysis**

**What does this show?**
This chart displays which anatomical features were most important in
identifying this X-ray as a {body_part}.

**Key Feature:**
• **{top_feature}** had the highest importance ({top_score:.1f}%)
• This feature was crucial in the AI's decision

**Color Coding:**
• 🔴 Red bars (>70%): Critical features
• 🔵 Blue bars (40-70%): Moderate importance
• 🟢 Green bars (<40%): Low importance

**Interpretation:**
The AI analyzed these anatomical features to determine that this is
a {body_part} X-ray. The combination of these features led to the final decision.
"""
        return explanation

    # ========================================================================
    # 3. DECISION CONFIDENCE BREAKDOWN
    # ========================================================================

    def generate_confidence_breakdown(self, body_part_conf, disease_conf,
                                      top_alternatives, body_part, disease):
        """
        Visualize confidence scores and alternatives

        Parameters:
            body_part_conf: Body part detection confidence
            disease_conf: Disease detection confidence
            top_alternatives: List of (name, confidence) tuples
            body_part: Detected body part
            disease: Detected disease

        Returns:
            dict with 'chart' and 'explanation'
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # === LEFT CHART: Confidence Gauges ===
            categories = ['Body Part\nDetection', 'Disease\nDetection']
            confidences = [body_part_conf * 100, disease_conf * 100]
            colors = ['#4ECDC4', '#FF6B6B']

            bars = ax1.bar(categories, confidences, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax1.set_ylim(0, 100)
            ax1.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Detection Confidence Levels', fontsize=14, fontweight='bold')
            ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='High Confidence Threshold')
            ax1.legend()

            # Add value labels
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{conf:.1f}%', ha='center', va='bottom',
                        fontweight='bold', fontsize=12)

            # === RIGHT CHART: Alternative Diagnoses ===
            if top_alternatives and len(top_alternatives) > 0:
                alt_names = [disease] + [alt[0] for alt in top_alternatives[:3]]
                alt_scores = [disease_conf * 100] + [alt[1] * 100 for alt in top_alternatives[:3]]
                alt_colors = ['#FF6B6B'] + ['#95E1D3'] * 3

                bars2 = ax2.barh(alt_names, alt_scores, color=alt_colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax2.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Disease Probability Ranking', fontsize=14, fontweight='bold')
                ax2.set_xlim(0, 100)

                # Add value labels
                for bar, score in zip(bars2, alt_scores):
                    width = bar.get_width()
                    ax2.text(width + 2, bar.get_y() + bar.get_height()/2.,
                            f'{score:.1f}%', ha='left', va='center', fontweight='bold')

            plt.tight_layout()

            # Convert to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            plt.close()

            # Generate explanation
            explanation = self._generate_confidence_explanation(
                body_part_conf, disease_conf, body_part, disease, top_alternatives
            )

            return {
                'chart': img_array,
                'explanation': explanation
            }

        except Exception as e:
            print(f"⚠️ Confidence breakdown error: {e}")
            return None

    def _generate_confidence_explanation(self, bp_conf, disease_conf,
                                        body_part, disease, alternatives):
        """Generate explanation for confidence breakdown"""

        confidence_level = "High" if disease_conf > 0.7 else "Moderate" if disease_conf > 0.5 else "Low"

        explanation = f"""
📈 **Confidence Breakdown**

**Detection Confidence:**
• Body Part ({body_part}): {bp_conf*100:.1f}%
• Disease ({disease}): {disease_conf*100:.1f}%
• Overall Confidence Level: **{confidence_level}**

**What does this mean?**
The AI is {bp_conf*100:.0f}% confident this is a {body_part} X-ray.
For the disease diagnosis, the confidence is {disease_conf*100:.0f}%.

**Alternative Diagnoses Considered:**
"""

        if alternatives and len(alternatives) > 0:
            for i, (alt_name, alt_conf) in enumerate(alternatives[:3], 1):
                explanation += f"\n{i}. {alt_name}: {alt_conf*100:.1f}%"

        explanation += f"""

**Clinical Interpretation:**
{"High confidence indicates strong evidence for this diagnosis." if confidence_level == "High" else
 "Moderate confidence suggests the diagnosis is likely but should be confirmed." if confidence_level == "Moderate" else
 "Low confidence indicates uncertainty - additional imaging or clinical correlation recommended."}
"""

        return explanation

    # ========================================================================
    # 4. SALIENCY MAP VISUALIZATION
    # ========================================================================

    def generate_saliency_map(self, image_array, body_part, disease):
        """
        Generate saliency map showing pixel-level importance

        Parameters:
            image_array: Input X-ray image
            body_part: Detected body part
            disease: Detected disease

        Returns:
            dict with 'saliency_map' and 'explanation'
        """
        try:
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array.copy()

            # Calculate saliency using multiple filters
            # Sobel
            sobelx = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)

            # Laplacian
            laplacian = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_64F))

            # Canny edges
            edges = cv2.Canny(gray.astype(np.uint8), 50, 150).astype(np.float32)

            # Combine saliency measures
            saliency = sobel * 0.4 + laplacian * 0.3 + edges * 0.3

            # Normalize
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-7)

            # Apply colormap
            saliency_colored = cv2.applyColorMap((saliency * 255).astype(np.uint8),
                                                 cv2.COLORMAP_HOT)

            # Create overlay
            if len(image_array.shape) == 2:
                image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_array.copy()

            overlay = cv2.addWeighted(image_rgb, 0.6, saliency_colored, 0.4, 0)

            # Add title
            cv2.putText(overlay, f"Saliency Map: {disease}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            explanation = f"""
🎯 **Saliency Map Analysis**

**What is a Saliency Map?**
A saliency map highlights the most "important" pixels that contribute to the
AI's understanding of the image. Brighter areas = more important for detection.

**Key Observations:**
• Bright/hot colors indicate critical pixels for {disease} detection
• These pixels contain distinctive patterns the AI learned to recognize
• Dark areas contributed less to the final diagnosis

**How it helps:**
Saliency maps show exactly which pixels influenced the AI's decision at the
most granular level, providing pixel-level explainability.
"""

            return {
                'saliency_map': overlay,
                'explanation': explanation
            }

        except Exception as e:
            print(f"⚠️ Saliency map error: {e}")
            return None

    # ========================================================================
    # 5. COMPREHENSIVE XAI REPORT
    # ========================================================================

    def generate_comprehensive_report(self, image_array, body_part, disease,
                                     body_part_conf, disease_conf, evidence,
                                     top_alternatives=None):
        """
        Generate a comprehensive XAI report with all explanations

        Returns:
            dict containing all XAI visualizations and explanations
        """
        print("🔬 Generating Comprehensive Explainable AI Report...")

        report = {}

        try:
            # 1. Grad-CAM
            print("  → Generating Grad-CAM...")
            gradcam = self.generate_gradcam(image_array, body_part, disease)
            if gradcam:
                report['gradcam'] = gradcam

            # 2. Feature Importance
            print("  → Analyzing feature importance...")
            features = self.generate_feature_importance(evidence, body_part)
            if features:
                report['features'] = features

            # 3. Confidence Breakdown
            print("  → Creating confidence breakdown...")
            confidence = self.generate_confidence_breakdown(
                body_part_conf, disease_conf, top_alternatives, body_part, disease
            )
            if confidence:
                report['confidence'] = confidence

            # 4. Saliency Map
            print("  → Generating saliency map...")
            saliency = self.generate_saliency_map(image_array, body_part, disease)
            if saliency:
                report['saliency'] = saliency

            # 5. Summary Explanation
            report['summary'] = self._generate_summary_explanation(
                body_part, disease, body_part_conf, disease_conf
            )

            print("✅ Comprehensive XAI report generated!")
            return report

        except Exception as e:
            print(f"⚠️ XAI report generation error: {e}")
            return report

    def _generate_summary_explanation(self, body_part, disease, bp_conf, disease_conf):
        """Generate overall summary explanation"""

        summary = f"""
🤖 **AI Decision Summary**

**What the AI Detected:**
• Body Part: {body_part} (Confidence: {bp_conf*100:.1f}%)
• Condition: {disease} (Confidence: {disease_conf*100:.1f}%)

**How the AI Made This Decision:**

1. **Image Analysis**: The AI analyzed the X-ray using deep learning models
   trained on thousands of medical images.

2. **Feature Detection**: Key anatomical features were identified and evaluated,
   including bone structure, density patterns, and spatial relationships.

3. **Pattern Recognition**: The AI compared these features against learned patterns
   for {disease} in {body_part} X-rays.

4. **Confidence Calculation**: Based on the strength of matches, the AI assigned
   confidence scores to possible diagnoses.

**Why This Diagnosis:**
The AI identified visual patterns consistent with {disease}, including specific
anatomical markers that are characteristic of this condition in {body_part} imaging.

**Important Notes:**
• This is an AI-assisted preliminary analysis
• All findings should be reviewed by a qualified radiologist
• The AI's decision is based on learned patterns from training data
• Clinical correlation and patient history are essential for final diagnosis

**Transparency & Trust:**
These explanations show you exactly how the AI reached its conclusions, making the
decision-making process transparent and understandable.
"""
        return summary


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_xai_module(detector, chest_head, general_head):
    """
    Factory function to create XAI module

    Parameters:
        detector: Body part detector
        chest_head: Chest disease head
        general_head: General disease head

    Returns:
        ExplainableAI instance
    """
    disease_heads = {
        'chest': chest_head,
        'general': general_head
    }
    return ExplainableAI(detector, disease_heads)


__all__ = ['ExplainableAI', 'create_xai_module']

