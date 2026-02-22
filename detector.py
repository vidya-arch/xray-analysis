
import numpy as np
import cv2
import torch
from PIL import Image
import open_clip
from skimage import morphology, measure
from scipy import ndimage
from enhanced_heatmap_generator import HeatmapGenerator
from enhanced_feature_detector import AnatomicalFeatureDetector
from comprehensive_output import ComprehensiveOutputGenerator

# ============================================================================
# GLOBAL BODY PART LIST (USED EVERYWHERE)
# ============================================================================

ALL_BODY_PARTS = [
    'Fingers', 'Hand', 'Shoulder', 'Neck', 'Skull', 'Spine',
    'Knee', 'Ankle', 'Foot', 'Pelvis', 'Jaw', 'Chest',
    'Wrist', 'Elbow', 'Abdomen', 'Thigh'
]

# ============================================================================
# ENHANCED CLIP-BASED DETECTOR
# ============================================================================

class EnhancedCLIPXRayDetector:
    def __init__(self):
        print("🔄 Loading CLIP model...")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',
                pretrained='laion2b_s34b_b79k'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"⚠ CLIP failed to load: {e}")
            self.model = None

    def detect_with_clip(self, image_pil):
        if self.model is None:
            return None

        print("\n🔬 ENHANCED CLIP ZERO-SHOT CLASSIFICATION:")

        text_prompts = [
            "a medical X-ray radiograph showing multiple finger phalanges and interphalangeal joints", # Fingers
            "a medical X-ray radiograph of a complete hand showing all five fingers, metacarpal bones, and wrist", # Hand
            "a medical X-ray radiograph of shoulder joint showing humerus, scapula, and clavicle bones", # Shoulder
            "a medical X-ray radiograph of cervical spine showing neck vertebrae C1 through C7", # Neck
            "a medical X-ray radiograph of human skull and cranium showing frontal and lateral bone structure", # Skull
            "a medical X-ray radiograph of thoracic or lumbar spine showing multiple vertebrae in vertical alignment", # Spine
            "a medical X-ray radiograph of knee joint showing femur, tibia, fibula, and patella", # Knee
            "a medical X-ray radiograph of ankle joint showing tibia, fibula, and talus bones", # Ankle
            "a medical X-ray radiograph of foot showing toe phalanges, metatarsal bones, and tarsal bones", # Foot
            "a medical X-ray radiograph of pelvis and hip bones showing bilateral iliac wings and femoral heads", # Pelvis
            "a medical X-ray radiograph of mandible and jaw showing teeth and lower facial bones", # Jaw
            "a medical chest X-ray radiograph showing bilateral lung fields, ribs, and thoracic cavity", # Chest
            "a medical X-ray radiograph of wrist showing carpal bones, radius, and ulna", # Wrist
            "a medical X-ray radiograph of elbow joint showing humerus, radius, and ulna articulation", # Elbow
            "a medical abdominal X-ray radiograph showing bowel gas pattern and soft tissue structures", # Abdomen
            "a medical X-ray radiograph of femur showing the long thigh bone shaft" # Thigh
        ]

        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(text_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            temperature = 0.01
            similarity = (100.0 * image_features @ text_features.T / temperature).softmax(dim=-1)
            similarity = similarity.cpu().numpy()[0]

        scores = {part: float(sim * 100) for part, sim in zip(ALL_BODY_PARTS, similarity)}
        sorted_results = sorted(zip(ALL_BODY_PARTS, similarity), key=lambda x: x[1], reverse=True)
        print("  Top 5 CLIP results:")
        for part, score in sorted_results[:5]:
            print(f"    {part}: {score*100:.1f}%")

        return scores


# ============================================================================
# ENHANCED ANATOMICAL FEATURE DETECTOR
# ============================================================================

class EnhancedAnatomicalDetector:
    @staticmethod
    def detect_features(image_array):
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape
        aspect_ratio = w / h

        print("🔬 ENHANCED ANATOMICAL FEATURE DETECTION:")

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        thresh = np.percentile(enhanced, 65)
        bone_mask = enhanced > thresh
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=200)
        bone_mask = morphology.binary_closing(bone_mask, morphology.disk(2))

        labeled = measure.label(bone_mask, connectivity=2)
        regions = measure.regionprops(labeled)

        elongated_bones = []
        round_bones = []
        small_bones = []

        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            region_h = maxr - minr
            region_w = maxc - minc
            if region_w > 0 and region_h > 0:
                elongation = max(region_h, region_w) / min(region_h, region_w)
                circularity = 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
                if elongation > 3.0 and region.area > 400:
                    elongated_bones.append(region.area)
                if 0.6 < circularity < 1.0 and region.area > 1000:
                    round_bones.append(region.area)
                if 100 < region.area < 800:
                    small_bones.append(region.area)

        num_elongated = len(elongated_bones)
        num_round = len(round_bones)
        num_small = len(small_bones)

        left = enhanced[:, :w//2]
        right = cv2.flip(enhanced[:, w//2:], 1)
        min_w = min(left.shape[1], right.shape[1])
        if min_w > 0:
            left_crop = left[:, :min_w]
            right_crop = right[:, :min_w]
            correlation = np.corrcoef(left_crop.flatten(), right_crop.flatten())[0, 1]
            symmetry = max(0, correlation)
        else:
            symmetry = 0.0

        dark_thresh = np.percentile(enhanced, 25)
        very_dark_ratio = np.sum(enhanced < dark_thresh) / enhanced.size
        bright_thresh = np.percentile(enhanced, 85)
        very_bright_ratio = np.sum(enhanced > bright_thresh) / enhanced.size

        center_region = enhanced[h//3:2*h//3, w//3:2*w//3]
        peripheral_region = enhanced.copy()
        peripheral_region[h//3:2*h//3, w//3:2*w//3] = 0
        center_density = np.mean(center_region)
        peripheral_density = np.mean(peripheral_region[peripheral_region > 0]) if np.any(peripheral_region > 0) else 0
        central_concentration = center_density / (peripheral_density + 1e-6)

        edges = cv2.Canny(enhanced, 50, 150)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_line_density = np.sum(vertical_lines > 0) / vertical_lines.size

        scores = {part: 20 for part in ALL_BODY_PARTS}

        scores.update({
            'Fingers': 85 if 3 <= num_elongated <= 5 and aspect_ratio < 1.5 else 60 if 2 <= num_elongated <= 6 else 20,
            'Hand': 90 if 5 <= num_elongated <= 10 and num_small >= 3 else 70 if 4 <= num_elongated <= 12 else 25,
            'Chest': 95 if symmetry > 0.70 and very_dark_ratio > 0.20 and aspect_ratio > 0.9 else 70 if symmetry > 0.60 and very_dark_ratio > 0.15 else 20,
            'Foot': 75 if 4 <= num_elongated <= 8 and aspect_ratio > 0.8 else 50 if 3 <= num_elongated <= 10 else 20,
            'Ankle': 70 if 2 <= num_elongated <= 5 and 0.7 < aspect_ratio < 1.3 else 45 if 1 <= num_elongated <= 6 else 20,
            'Knee': 80 if num_round >= 1 and 1 <= num_elongated <= 4 else 55 if num_round >= 1 or num_elongated <= 5 else 20,
            'Pelvis': 88 if symmetry > 0.65 and num_round >= 2 else 60 if symmetry > 0.50 and num_round >= 1 else 20,
            'Shoulder': 75 if num_round >= 1 and 0.9 < aspect_ratio < 1.3 else 50 if num_round >= 1 else 20,
            'Skull': 80 if symmetry > 0.55 and central_concentration > 1.2 else 55 if symmetry > 0.45 else 20,
            'Spine': 85 if vertical_line_density > 0.02 and num_elongated >= 2 else 60 if vertical_line_density > 0.015 else 20,
            'Neck': 70 if vertical_line_density > 0.015 and num_elongated <= 3 else 50 if vertical_line_density > 0.01 else 20,
            'Jaw': 65 if 0.8 < aspect_ratio < 1.5 and num_small >= 2 else 40,
            'Wrist': 75 if num_small >= 4 and aspect_ratio < 1.2 else 50 if num_small >= 2 else 20,
            'Elbow': 70 if num_small >= 2 and 1.0 < aspect_ratio < 1.5 else 45,
            'Abdomen': 60 if very_dark_ratio > 0.25 and symmetry > 0.50 else 35,
            'Thigh': 65 if num_elongated == 1 and aspect_ratio > 1.5 else 40
        })

        features = {
            'elongated_bones': num_elongated,
            'round_bones': num_round,
            'small_bones': num_small,
            'symmetry': symmetry,
            'aspect_ratio': aspect_ratio,
            'dark_ratio': very_dark_ratio,
            'bright_ratio': very_bright_ratio,
            'central_concentration': central_concentration,
            'vertical_lines': vertical_line_density
        }

        print(f"  Elongated bones: {num_elongated}, Round bones: {num_round}, Small bones: {num_small}")
        print(f"  Symmetry: {symmetry:.0%}, Aspect: {aspect_ratio:.2f}, Vertical: {vertical_line_density:.3f}")

        return scores, features


# ============================================================================
# IMPROVED HYBRID DETECTOR (COMBINING CLIP + ANATOMICAL)
# ============================================================================

class ImprovedHybridDetector:
    def __init__(self):
        print("🔧 Initializing Hybrid Detector...")
        self.clip_detector = EnhancedCLIPXRayDetector()
        self.anatomical_detector = EnhancedAnatomicalDetector()
        print("✓ Hybrid Detector Ready\n")
        self.heatmap_gen = HeatmapGenerator(self.clip_detector)
        self.feature_det = AnatomicalFeatureDetector()
        self.output_gen = ComprehensiveOutputGenerator(
            self.heatmap_gen,
            self.feature_det
        )

    def generate_enhanced_outputs(self, image_array, body_part, diseases):
        """Generate all enhanced outputs"""
        results = self.output_gen.generate_complete_analysis(
            image_array,
            body_part,
            diseases
        )
        return results
    def detect(self, image_array):
        print("\n" + "="*70)
        print("🔍 STARTING BODY PART DETECTION")
        print("="*70)

        image_pil = Image.fromarray(image_array).convert('RGB')

        clip_scores = self.clip_detector.detect_with_clip(image_pil)
        anatomical_scores, features = self.anatomical_detector.detect_features(image_array)

        if clip_scores is None:
            print("⚠ CLIP unavailable, using anatomical features only")
            final_scores = anatomical_scores
        else:
            final_scores = {}
            for part in ALL_BODY_PARTS:
                clip_weight = 0.70
                anat_weight = 0.30
                final_scores[part] = (
                    clip_weight * clip_scores.get(part, 0) +
                    anat_weight * anatomical_scores.get(part, 0)
                )

        sorted_parts = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        print("\n📊 FINAL COMBINED SCORES (Top 5):")
        for part, score in sorted_parts[:5]:
            print(f"  {part}: {score:.1f} pts")

        best_part = sorted_parts[0][0]
        best_score = sorted_parts[0][1]
        confidence = min(best_score / 100.0, 1.0)

        evidence = {
            'final_scores': final_scores,
            'clip_scores': clip_scores if clip_scores else {},
            'anatomical_scores': anatomical_scores,
            'features': features
        }

        print(f"\n✅ FINAL DETECTION: {best_part} (Confidence: {confidence*100:.1f}%)")
        print("="*70 + "\n")

        return best_part, confidence, evidence


__all__ = ["ImprovedHybridDetector", "ALL_BODY_PARTS"]
