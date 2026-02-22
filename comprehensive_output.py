


import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
import json


class ComprehensiveOutputGenerator:
    """
    Generates all enhanced outputs including:
    - Pixel-level attention heatmap
    - Bounding box detection
    - Disease probability scores
    - Detailed anatomical features
    """

    def __init__(self, heatmap_generator, feature_detector):
        """
        Parameters
        ----------
        heatmap_generator : HeatmapGenerator instance
        feature_detector : AnatomicalFeatureDetector instance
        """
        self.heatmap_gen = heatmap_generator
        self.feature_det = feature_detector

    def generate_complete_analysis(self, image_array, body_part, diseases,
                                   anatomical_scores=None):
        """
        Generate comprehensive analysis with all outputs.

        Parameters
        ----------
        image_array : numpy array
            Input X-ray image
        body_part : str
            Detected body part
        diseases : list of (name, probability) tuples
            Disease predictions
        anatomical_scores : dict, optional
            Pre-computed anatomical feature scores

        Returns
        -------
        dict with keys:
            - 'heatmap_overlay': RGB image with heatmap overlay
            - 'bounding_boxes': List of detected abnormal regions
            - 'disease_probabilities': Formatted disease scores
            - 'anatomical_features': Detailed feature measurements
            - 'visualization': Combined visualization image
            - 'summary_text': Human-readable summary
        """
        print("\n" + "="*70)
        print("🔬 GENERATING COMPREHENSIVE ANALYSIS")
        print("="*70)

        # 1. Generate heatmap and bounding boxes
        print("\n[1/4] Generating attention heatmap...")
        heatmap_result = self.heatmap_gen.generate(image_array, body_part, diseases)

        # 2. Get detailed anatomical features
        print("\n[2/4] Analyzing anatomical features...")
        if anatomical_scores is None:
            anatomical_scores, anatomical_features = self.feature_det.detect(image_array)
        else:
            _, anatomical_features = self.feature_det.detect(image_array)

        # 3. Format disease probabilities
        print("\n[3/4] Formatting disease predictions...")
        disease_probs = self._format_disease_probabilities(diseases)

        # 4. Create combined visualization
        print("\n[4/4] Creating visualization...")
        visualization = self._create_combined_visualization(
            image_array,
            heatmap_result,
            body_part,
            diseases,
            anatomical_features
        )

        # Generate summary text
        summary = self._generate_summary_text(
            body_part,
            diseases,
            heatmap_result['bboxes'],
            anatomical_features
        )

        result = {
            'heatmap_overlay': heatmap_result['heatmap'],
            'bounding_boxes': heatmap_result['bboxes'],
            'raw_heatmap': heatmap_result['raw_heatmap'],
            'disease_probabilities': disease_probs,
            'anatomical_features': anatomical_features,
            'visualization': visualization,
            'summary_text': summary,
            'json_export': self._create_json_export(
                body_part, diseases, heatmap_result['bboxes'], anatomical_features
            )
        }

        print("\n✅ Complete analysis generated successfully!")
        print("="*70 + "\n")

        return result

    def _format_disease_probabilities(self, diseases):
        """Format disease predictions with visual probability bars"""
        formatted = []
        formatted.append("🏥 DISEASE PROBABILITY ANALYSIS")
        formatted.append("=" * 60)

        for i, (disease_name, probability) in enumerate(diseases[:5], 1):
            # Create visual probability bar
            bar_length = 30
            filled = int(probability * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Color coding
            if probability > 0.7:
                status = "🔴 HIGH"
            elif probability > 0.4:
                status = "🟡 MODERATE"
            else:
                status = "🟢 LOW"

            formatted.append(f"\n{i}. {disease_name}")
            formatted.append(f"   Probability: {probability*100:.1f}% {status}")
            formatted.append(f"   [{bar}]")

        formatted.append("\n" + "=" * 60)
        return "\n".join(formatted)

    def _create_combined_visualization(self, image_array, heatmap_result,
                                      body_part, diseases, anatomical_features):
        """
        Create a comprehensive visualization combining multiple outputs.
        """
        heatmap_img = heatmap_result['heatmap']
        h, w = heatmap_img.shape[:2]

        # Create a larger canvas for the combined visualization
        # Layout: Original | Heatmap | Info Panel
        canvas_width = w * 3
        canvas_height = h
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 1. Original image (left)
        if len(image_array.shape) == 2:
            original_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            original_rgb = image_array.copy()
        if original_rgb.dtype != np.uint8:
            original_rgb = ((original_rgb * 255) if original_rgb.max() <= 1.0
                           else original_rgb).astype(np.uint8)

        # Draw bounding boxes on original
        original_with_boxes = original_rgb.copy()
        for idx, (x, y, bbox_w, bbox_h, conf) in enumerate(heatmap_result['bboxes']):
            color = (0, 255, 0)  # Green boxes on original
            thickness = max(2, int(min(w, h) / 300))
            cv2.rectangle(original_with_boxes, (x, y),
                         (x + bbox_w, y + bbox_h), color, thickness)

            label = f"Region {idx+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.4
            cv2.putText(original_with_boxes, label, (x, y - 5),
                       font, fs, color, 1, cv2.LINE_AA)

        canvas[:h, :w] = original_with_boxes

        # 2. Heatmap overlay (center)
        canvas[:h, w:2*w] = heatmap_img

        # 3. Info panel (right)
        info_panel = np.zeros((h, w, 3), dtype=np.uint8)

        # Add text information to info panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        line_height = 25

        # Title
        cv2.putText(info_panel, "ANALYSIS REPORT", (10, y_offset),
                   font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += line_height * 2

        # Body part
        cv2.putText(info_panel, f"Body Part: {body_part}", (10, y_offset),
                   font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
        y_offset += line_height

        # Primary disease
        if diseases:
            primary = diseases[0]
            cv2.putText(info_panel, f"Finding: {primary[0]}", (10, y_offset),
                       font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
            y_offset += line_height
            cv2.putText(info_panel, f"Confidence: {primary[1]*100:.1f}%", (10, y_offset),
                       font, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
            y_offset += line_height * 2

        # Bounding boxes info
        cv2.putText(info_panel, "Detected Regions:", (10, y_offset),
                   font, 0.5, (255, 255, 100), 1, cv2.LINE_AA)
        y_offset += line_height

        for idx, (x, y, bbox_w, bbox_h, conf) in enumerate(heatmap_result['bboxes'][:3], 1):
            text = f"  {idx}. Size: {bbox_w}x{bbox_h}, Conf: {conf*100:.0f}%"
            cv2.putText(info_panel, text, (10, y_offset),
                       font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += line_height

        y_offset += line_height

        # Anatomical features
        cv2.putText(info_panel, "Anatomical Features:", (10, y_offset),
                   font, 0.5, (255, 255, 100), 1, cv2.LINE_AA)
        y_offset += line_height

        features_text = [
            f"  Bones: {anatomical_features['elongated_bones_count']}",
            f"  Symmetry: {anatomical_features['bilateral_symmetry_percent']:.0f}%",
            f"  Density: {anatomical_features['bone_density_percent']:.1f}%",
        ]

        for text in features_text:
            cv2.putText(info_panel, text, (10, y_offset),
                       font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
            y_offset += line_height

        canvas[:h, 2*w:] = info_panel

        # Add labels to each section
        cv2.putText(canvas, "ORIGINAL + BOXES", (10, 20),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ATTENTION HEATMAP", (w + 10, 20),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas

    def _generate_summary_text(self, body_part, diseases, bboxes, features):
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 70)
        lines.append("COMPREHENSIVE X-RAY ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append("")

        # Body part info
        lines.append(f"🎯 BODY PART: {body_part}")
        lines.append("")

        # Disease findings
        lines.append("🏥 FINDINGS:")
        if diseases:
            for i, (name, prob) in enumerate(diseases[:3], 1):
                confidence = "HIGH" if prob > 0.7 else "MODERATE" if prob > 0.4 else "LOW"
                lines.append(f"  {i}. {name}: {prob*100:.1f}% ({confidence} confidence)")
        lines.append("")

        # Abnormal regions
        lines.append(f"📦 DETECTED ABNORMAL REGIONS: {len(bboxes)}")
        for i, (x, y, w, h, conf) in enumerate(bboxes, 1):
            lines.append(f"  Region {i}: Position ({x}, {y}), "
                        f"Size {w}×{h}px, Confidence {conf*100:.0f}%")
        lines.append("")

        # Key anatomical features
        lines.append("🔬 KEY ANATOMICAL FEATURES:")
        lines.append(f"  • Bone structures: {features['elongated_bones_count']} elongated, "
                    f"{features['round_structures_count']} round")
        lines.append(f"  • Bilateral symmetry: {features['bilateral_symmetry_percent']:.1f}%")
        lines.append(f"  • Bone density: {features['bone_density_percent']:.1f}%")
        lines.append(f"  • Pattern: {features['primary_bone_pattern']}")
        lines.append(f"  • Spatial distribution: {features['spatial_distribution']}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _create_json_export(self, body_part, diseases, bboxes, features):
        """Create JSON export of all data"""
        return {
            "body_part": body_part,
            "diseases": [
                {"name": name, "probability": float(prob)}
                for name, prob in diseases
            ],
            "abnormal_regions": [
                {
                    "region_id": i + 1,
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "confidence": float(conf)
                }
                for i, (x, y, w, h, conf) in enumerate(bboxes)
            ],
            "anatomical_features": {
                "bone_count": {
                    "elongated": features['elongated_bones_count'],
                    "round": features['round_structures_count']
                },
                "symmetry": {
                    "bilateral_percent": round(features['bilateral_symmetry_percent'], 2),
                    "correlation": round(features['symmetry_correlation'], 2)
                },
                "density": {
                    "bone_density_percent": round(features['bone_density_percent'], 2),
                    "dark_area_percent": round(features['dark_area_ratio_percent'], 2)
                },
                "image_properties": {
                    "dimensions": features['image_dimensions'],
                    "aspect_ratio": round(features['aspect_ratio'], 2),
                    "edge_density": round(features['edge_density'], 4),
                    "texture_complexity": round(features['texture_complexity'], 2)
                },
                "interpretation": {
                    "bone_pattern": features['primary_bone_pattern'],
                    "symmetry_assessment": features['symmetry_assessment'],
                    "spatial_distribution": features['spatial_distribution']
                }
            }
        }

    def save_outputs(self, output_dir, results, prefix="xray_analysis"):
        """
        Save all outputs to files.

        Parameters
        ----------
        output_dir : str
            Directory to save outputs
        results : dict
            Output from generate_complete_analysis()
        prefix : str
            Prefix for output filenames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save visualization
        viz_path = os.path.join(output_dir, f"{prefix}_visualization.png")
        cv2.imwrite(viz_path, cv2.cvtColor(results['visualization'], cv2.COLOR_RGB2BGR))
        print(f"✓ Saved visualization: {viz_path}")

        # Save heatmap overlay
        heatmap_path = os.path.join(output_dir, f"{prefix}_heatmap.png")
        cv2.imwrite(heatmap_path, cv2.cvtColor(results['heatmap_overlay'], cv2.COLOR_RGB2BGR))
        print(f"✓ Saved heatmap: {heatmap_path}")

        # Save summary text
        summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(results['summary_text'])
            f.write("\n\n")
            f.write(results['disease_probabilities'])
        print(f"✓ Saved summary: {summary_path}")

        # Save JSON export
        json_path = os.path.join(output_dir, f"{prefix}_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results['json_export'], f, indent=2)
        print(f"✓ Saved JSON data: {json_path}")

        return {
            'visualization': viz_path,
            'heatmap': heatmap_path,
            'summary': summary_path,
            'json': json_path
        }

