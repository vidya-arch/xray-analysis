
# [Copy entire content of enhanced_feature_detector.py here]


import numpy as np
import cv2
from skimage import morphology, measure


class AnatomicalFeatureDetector:
    """
    Detects anatomical features (bone count, symmetry, dark areas)
    and scores each body part accordingly.
    Enhanced with detailed anatomical measurements.
    """

    @staticmethod
    def detect(image_array):
        """
        Analyse *image_array* (numpy, RGB or grayscale).

        Returns
        -------
        scores : dict[str, float]
            Body-part name → heuristic score.
        features : dict
            Detailed feature measurements for the report.
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array

        h, w = gray.shape
        print("🔬 ANATOMICAL FEATURE DETECTION:")

        # ---- Enhance ----
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # ---- Feature 1: Elongated bone count ----
        thresh = np.percentile(enhanced, 70)
        bone_mask = enhanced > thresh
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=300)

        labeled = measure.label(bone_mask, connectivity=2)
        regions = measure.regionprops(labeled)

        elongated_bones = []
        round_structures = []
        total_bone_area = 0

        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            region_h = maxr - minr
            region_w = maxc - minc
            if region_w > 0:
                elongation = max(region_h, region_w) / min(region_h, region_w)

                # Calculate circularity
                if region.perimeter > 0:
                    circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
                else:
                    circularity = 0

                total_bone_area += region.area

                if elongation > 2.5 and region.area > 500:
                    elongated_bones.append({
                        "elongation": elongation,
                        "area": region.area,
                        "length": max(region_h, region_w),
                        "width": min(region_h, region_w),
                        "position": (minc + region_w//2, minr + region_h//2)
                    })

                if circularity > 0.6 and region.area > 800:
                    round_structures.append({
                        "circularity": circularity,
                        "area": region.area,
                        "position": (minc + region_w//2, minr + region_h//2)
                    })

        num_elongated = len(elongated_bones)
        num_round = len(round_structures)
        bone_density = total_bone_area / (h * w)

        print(f"  Elongated bones: {num_elongated}")
        print(f"  Round structures: {num_round}")
        print(f"  Bone density: {bone_density:.2%}")

        # ---- Feature 2: Bilateral symmetry ----
        left = enhanced[:, : w // 2]
        right = cv2.flip(enhanced[:, w // 2 :], 1)
        min_w = min(left.shape[1], right.shape[1])

        if min_w > 0:
            diff = np.mean(
                np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
            )
            symmetry = 1.0 - (diff / 128.0)

            # Also calculate correlation-based symmetry
            correlation = np.corrcoef(
                left[:, :min_w].flatten(),
                right[:, :min_w].flatten()
            )[0, 1]
            symmetry_correlation = max(0, correlation)
        else:
            symmetry = 0.0
            symmetry_correlation = 0.0

        print(f"  Bilateral symmetry: {symmetry:.2%}")
        print(f"  Symmetry correlation: {symmetry_correlation:.2%}")

        # ---- Feature 3: Large dark areas (lungs/air spaces) ----
        dark_thresh = np.percentile(enhanced, 30)

        h_left = enhanced[:, : w // 3]
        h_right = enhanced[:, 2 * w // 3 :]
        h_left_dark = np.sum(h_left < dark_thresh) / h_left.size
        h_right_dark = np.sum(h_right < dark_thresh) / h_right.size

        v_top = enhanced[: h // 3, :]
        v_bottom = enhanced[2 * h // 3 :, :]
        v_top_dark = np.sum(v_top < dark_thresh) / v_top.size
        v_bottom_dark = np.sum(v_bottom < dark_thresh) / v_bottom.size

        has_bilateral_dark = (h_left_dark > 0.30 and h_right_dark > 0.30) or (
            v_top_dark > 0.30 and v_bottom_dark > 0.30
        )

        dark_area_ratio = np.sum(enhanced < dark_thresh) / enhanced.size

        print(f"  Bilateral dark areas (lungs): {has_bilateral_dark}")
        print(f"  Total dark area ratio: {dark_area_ratio:.2%}")

        # ---- Feature 4: Edge complexity ----
        edges = cv2.Canny(enhanced, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # ---- Feature 5: Texture analysis ----
        # Calculate local intensity variance
        kernel_size = 15
        mean_filter = cv2.blur(enhanced.astype(float), (kernel_size, kernel_size))
        variance = cv2.blur((enhanced.astype(float) - mean_filter) ** 2, (kernel_size, kernel_size))
        texture_complexity = np.mean(variance)

        # ---- Feature 6: Aspect ratio and orientation ----
        aspect = w / h

        print(f"  Edge density: {edge_density:.3f}")
        print(f"  Texture complexity: {texture_complexity:.1f}")
        print(f"  Aspect ratio: {aspect:.2f}")

        # ---- Scoring ----
        scores = {}

        scores["Fingers"] = 80 if 3 <= num_elongated <= 5 else 25

        if 4 <= num_elongated <= 8:
            scores["Hand"] = 85
        elif 3 <= num_elongated <= 9:
            scores["Hand"] = 60
        else:
            scores["Hand"] = 30

        chest_score = 0
        if has_bilateral_dark:
            chest_score += 70
        if symmetry > 0.65:
            chest_score += 20
        scores["Chest"] = chest_score if chest_score > 0 else 25

        if 3 <= num_elongated <= 6:
            scores["Foot"] = 50
            scores["Ankle"] = 45
        else:
            scores["Foot"] = 20
            scores["Ankle"] = 20

        if aspect < 0.7 and num_elongated < 10:
            scores["Spine"] = 60
            scores["Neck"] = 55
        else:
            scores["Spine"] = 20
            scores["Neck"] = 20

        scores["Skull"] = 65 if (symmetry > 0.75 and num_elongated < 5) else 20
        scores["Jaw"] = 45 if (0.6 < symmetry < 0.8 and num_elongated < 8) else 15

        if 2 <= num_elongated <= 6 and 0.5 < symmetry < 0.75:
            scores["Knee"] = 50
        else:
            scores["Knee"] = 20

        scores["Pelvis"] = 65 if (symmetry > 0.70 and aspect > 1.2) else 20

        if symmetry < 0.6 and 3 <= num_elongated <= 10:
            scores["Shoulder"] = 50
        else:
            scores["Shoulder"] = 20

        scores["Wrist"] = 45 if 6 <= num_elongated <= 12 else 15
        scores["Elbow"] = 40 if 2 <= num_elongated <= 6 else 15
        scores["Abdomen"] = 45 if num_elongated < 3 else 15
        scores["Thigh"] = 40 if 1 <= num_elongated <= 3 else 15

        print(
            f"  Feature scores: Hand={scores['Hand']}, "
            f"Chest={scores['Chest']}, Fingers={scores.get('Fingers', 0)}\n"
        )

        # ---- Detailed features for output ----
        features = {
            # Bone structure
            "elongated_bones_count": num_elongated,
            "round_structures_count": num_round,
            "bone_density_percent": bone_density * 100,
            "elongated_bones_details": elongated_bones[:5],  # Top 5

            # Symmetry measurements
            "bilateral_symmetry_percent": symmetry * 100,
            "symmetry_correlation": symmetry_correlation * 100,
            "has_bilateral_dark_areas": has_bilateral_dark,

            # Spatial features
            "dark_area_ratio_percent": dark_area_ratio * 100,
            "left_dark_percent": h_left_dark * 100,
            "right_dark_percent": h_right_dark * 100,
            "top_dark_percent": v_top_dark * 100,
            "bottom_dark_percent": v_bottom_dark * 100,

            # Image characteristics
            "aspect_ratio": aspect,
            "edge_density": edge_density,
            "texture_complexity": texture_complexity,
            "image_dimensions": {"width": w, "height": h},

            # Interpretation - FIXED: Call static methods without 'self.'
            "primary_bone_pattern": AnatomicalFeatureDetector._interpret_bone_pattern(num_elongated, num_round),
            "symmetry_assessment": AnatomicalFeatureDetector._interpret_symmetry(symmetry, symmetry_correlation),
            "spatial_distribution": AnatomicalFeatureDetector._interpret_spatial(h_left_dark, h_right_dark, v_top_dark, v_bottom_dark)
        }

        return scores, features

    @staticmethod
    def _interpret_bone_pattern(elongated, round_count):
        """Interpret bone pattern for human-readable output"""
        if elongated >= 10:
            return "Complex multi-bone structure (Hand/Foot/Spine)"
        elif 5 <= elongated <= 9:
            return "Multiple elongated bones (Hand/Foot)"
        elif 3 <= elongated <= 4:
            return "Few elongated structures (Fingers/Toes)"
        elif 2 <= elongated:
            return "Paired bone structure (Arm/Leg)"
        else:
            return "Single or minimal bone structures"

    @staticmethod
    def _interpret_symmetry(symmetry, correlation):
        """Interpret symmetry measurements"""
        avg_symmetry = (symmetry + correlation) / 2
        if avg_symmetry > 0.75:
            return "High bilateral symmetry (Chest/Pelvis/Skull)"
        elif avg_symmetry > 0.60:
            return "Moderate symmetry (Chest/Spine)"
        elif avg_symmetry > 0.45:
            return "Low symmetry (Shoulder/irregular structures)"
        else:
            return "Asymmetric structure"

    @staticmethod
    def _interpret_spatial(left, right, top, bottom):
        """Interpret spatial distribution of dark areas"""
        if left > 0.30 and right > 0.30:
            return "Bilateral dark regions (likely lungs/air spaces)"
        elif top > 0.30 and bottom > 0.30:
            return "Vertical dark distribution"
        elif max(left, right, top, bottom) > 0.25:
            return "Localized dark region"
        else:
            return "Uniform density distribution"


def format_anatomical_features(features):
    """
    Format anatomical features for display in reports.

    Parameters
    ----------
    features : dict
        Output from AnatomicalFeatureDetector.detect()

    Returns
    -------
    str : Formatted text suitable for clinical reports
    """
    output = []
    output.append("=" * 60)
    output.append("DETAILED ANATOMICAL ANALYSIS")
    output.append("=" * 60)
    output.append("")

    # Bone Structure Section
    output.append("📊 BONE STRUCTURE:")
    output.append(f"  • Elongated bones detected: {features['elongated_bones_count']}")
    output.append(f"  • Round structures detected: {features['round_structures_count']}")
    output.append(f"  • Bone density: {features['bone_density_percent']:.1f}%")
    output.append(f"  • Pattern: {features['primary_bone_pattern']}")

    if features['elongated_bones_details']:
        output.append(f"  • Top elongated structures:")
        for i, bone in enumerate(features['elongated_bones_details'][:3], 1):
            output.append(f"    {i}. Length: {bone['length']}px, "
                         f"Elongation: {bone['elongation']:.1f}x")
    output.append("")

    # Symmetry Section
    output.append("🔄 BILATERAL SYMMETRY:")
    output.append(f"  • Symmetry score: {features['bilateral_symmetry_percent']:.1f}%")
    output.append(f"  • Correlation score: {features['symmetry_correlation']:.1f}%")
    output.append(f"  • Assessment: {features['symmetry_assessment']}")
    output.append("")

    # Spatial Distribution
    output.append("🗺️  SPATIAL DISTRIBUTION:")
    output.append(f"  • Dark area coverage: {features['dark_area_ratio_percent']:.1f}%")
    output.append(f"  • Left region: {features['left_dark_percent']:.1f}% dark")
    output.append(f"  • Right region: {features['right_dark_percent']:.1f}% dark")
    output.append(f"  • Pattern: {features['spatial_distribution']}")
    output.append("")

    # Image Characteristics
    output.append("📐 IMAGE CHARACTERISTICS:")
    dims = features['image_dimensions']
    output.append(f"  • Dimensions: {dims['width']} × {dims['height']} pixels")
    output.append(f"  • Aspect ratio: {features['aspect_ratio']:.2f}")
    output.append(f"  • Edge complexity: {features['edge_density']:.3f}")
    output.append(f"  • Texture complexity: {features['texture_complexity']:.1f}")
    output.append("")

    output.append("=" * 60)

    return "\n".join(output)

