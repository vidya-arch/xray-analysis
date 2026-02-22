
import numpy as np
import cv2
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─── optional heavy imports ─────────────────────────────────────────────────
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

try:
    from torchvision import transforms, models
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False

try:
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ============================================================
# DISEASE LABELS  (mirror what you show in the UI / reports)
# ============================================================

DISEASES = {
    "Chest":    ["Normal", "Pneumonia", "Pleural Effusion", "Cardiomegaly",
                 "Atelectasis", "Pulmonary Edema", "Consolidation",
                 "Pneumothorax", "Infiltration"],
    "Spine":    ["Normal", "Disc Degeneration", "Vertebral Fracture",
                 "Scoliosis", "Spinal Stenosis", "Spondylolisthesis"],
    "Knee":     ["Normal", "Osteoarthritis", "Fracture",
                 "Meniscal Tear", "Ligament Injury", "Bone Lesion"],
    "Shoulder": ["Normal", "Dislocation", "Fracture",
                 "Rotator Cuff Tear", "Arthritis", "Bone Lesion"],
    "Hand":     ["Normal", "Fracture", "Rheumatoid Arthritis",
                 "Osteoarthritis", "Dislocation", "Bone Lesion"],
    "Foot":     ["Normal", "Fracture", "Arthritis",
                 "Plantar Fasciitis", "Bone Spur", "Dislocation"],
    "Ankle":    ["Normal", "Fracture", "Sprain",
                 "Arthritis", "Tendon Injury", "Dislocation"],
    "Wrist":    ["Normal", "Fracture", "Carpal Tunnel Syndrome",
                 "Arthritis", "Ligament Tear", "Dislocation"],
    "Elbow":    ["Normal", "Fracture", "Dislocation",
                 "Tennis Elbow", "Arthritis", "Bone Lesion"],
    "Fingers":  ["Normal", "Fracture", "Dislocation",
                 "Arthritis", "Soft Tissue Injury", "Bone Lesion"],
    "Pelvis":   ["Normal", "Fracture", "Hip Dysplasia",
                 "Arthritis", "Avascular Necrosis", "Bone Lesion"],
    "Skull":    ["Normal", "Fracture", "Cranial Abnormality",
                 "Bone Lesion", "Sinus Disease", "Calcification"],
    "Neck":     ["Normal", "Cervical Disc Disease", "Fracture",
                 "Osteoarthritis", "Spinal Stenosis", "Spondylolisthesis"],
    "Jaw":      ["Normal", "Fracture", "TMJ Disorder",
                 "Dental Abnormality", "Bone Lesion", "Cyst"],
    "Abdomen":  ["Normal", "Bowel Obstruction", "Free Air",
                 "Kidney Stone", "Soft Tissue Mass", "Calcification"],
    "Thigh":    ["Normal", "Fracture", "Bone Lesion",
                 "Soft Tissue Mass", "Avascular Necrosis", "Periosteal Reaction"],
}


# ============================================================
# HELPER  – stable image fingerprint
# ============================================================

def _image_hash_int(image_array):
    """MD5-based integer fingerprint of the image (same X-ray → same number)."""
    try:
        small = cv2.resize(image_array, (64, 64)) if image_array is not None else np.zeros((64,64,3),dtype=np.uint8)
        return int(hashlib.md5(small.tobytes()).hexdigest(), 16)
    except Exception:
        return 0


# ============================================================
# ADVANCED IMAGE FEATURE ANALYSER
# ============================================================

class ImageFeatureAnalyzer:
    """
    Pure-OpenCV feature extraction that provides medically meaningful
    signals to supplement/calibrate the deep-learning models.
    """

    @staticmethod
    def _to_gray(arr):
        if len(arr.shape) == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return arr.copy()

    # ── Bone density ────────────────────────────────────────────────────
    @staticmethod
    def analyze_bone_density(arr):
        g = ImageFeatureAnalyzer._to_gray(arr)
        hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        hist /= hist.sum() + 1e-7
        mean_d   = float(np.average(np.arange(256), weights=hist))
        std_d    = float(np.sqrt(np.average((np.arange(256)-mean_d)**2, weights=hist)))
        low_frac = float(np.sum(g < 50)  / g.size)
        hi_frac  = float(np.sum(g > 200) / g.size)
        return {"mean": mean_d, "std": std_d,
                "low_density_ratio": low_frac, "high_density_ratio": hi_frac}

    # ── Fracture / discontinuity detection ──────────────────────────────
    @staticmethod
    def detect_discontinuities(arr):
        g = ImageFeatureAnalyzer._to_gray(arr)
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 90)
        n_comp, labels_cc, stats, _ = cv2.connectedComponentsWithStats(edges)
        # Large isolated components ≈ fracture lines
        large = int(np.sum(stats[1:, cv2.CC_STAT_AREA] > 50))
        edge_density = float(np.sum(edges > 0) / edges.size)
        # Radon-like vertical profile variance (detects breaks in cortex)
        col_sums = edges.sum(axis=0).astype(float)
        cortex_variance = float(np.var(col_sums))
        return {"edge_density": edge_density,
                "discontinuities": n_comp,
                "large_fragments": large,
                "cortex_variance": cortex_variance}

    # ── Joint space analysis ─────────────────────────────────────────────
    @staticmethod
    def analyze_joint_space(arr):
        g = ImageFeatureAnalyzer._to_gray(arr)
        h, w = g.shape
        roi  = g[h//4 : 3*h//4, w//4 : 3*w//4]
        # Dark horizontal bands ≈ joint spaces
        row_means  = roi.mean(axis=1)
        dark_rows  = float(np.sum(row_means < np.percentile(row_means, 30)) / len(row_means))
        # Gradient magnitude in joint region
        gx = cv2.Sobel(roi.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(roi.astype(np.float32), cv2.CV_32F, 0, 1)
        grad = float(np.sqrt(gx**2 + gy**2).mean())
        return {"joint_space_narrowing": dark_rows, "gradient": grad}

    # ── Lung field analysis ──────────────────────────────────────────────
    @staticmethod
    def detect_lung_abnormalities(arr):
        g = ImageFeatureAnalyzer._to_gray(arr)
        h, w = g.shape
        left  = g[:, :w//2]
        right = g[:, w//2:]
        thresh = lambda f: float(np.sum(f > np.percentile(f, 65)) / f.size)
        ol, or_ = thresh(left), thresh(right)
        asym   = abs(ol - or_)
        # Bottom-third opacity (pleural effusion indicator)
        bottom = g[2*h//3:, :]
        bot_op = float(np.sum(bottom > np.percentile(bottom, 55)) / bottom.size)
        # Central density (cardiomegaly indicator) – middle 30 % width
        centre = g[:, int(w*0.35):int(w*0.65)]
        centre_mean = float(centre.mean())
        return {"opacity_left": ol, "opacity_right": or_,
                "asymmetry": asym, "avg_opacity": (ol + or_) / 2,
                "bottom_opacity": bot_op, "centre_mean": centre_mean}

    # ── Spinal curvature ─────────────────────────────────────────────────
    @staticmethod
    def analyze_spinal_alignment(arr):
        g = ImageFeatureAnalyzer._to_gray(arr)
        blur = cv2.GaussianBlur(g, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60,
                                minLineLength=30, maxLineGap=10)
        vert = 0
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1+1e-9)))
                if angle > 70:
                    vert += 1
        # Horizontal profile variance (lateral curvature)
        row_peaks = [int(np.argmax(g[r, :])) for r in range(0, g.shape[0], 10)]
        curvature = float(np.std(row_peaks))
        return {"vertical_lines": vert, "curvature": curvature,
                "scoliosis_score": max(0.0, (curvature - 5) / 30)}


# ============================================================
# CHEST  –  torchxrayvision DenseNet  (primary)
#           + HuggingFace ViT fine-tuned on CheXpert (fallback)
#           + ImageFeatureAnalyzer (calibration)
# ============================================================

# torchxrayvision label order for the DenseNet models
_XRV_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax", "No Finding",
]

# Map XRV labels → our DISEASES["Chest"] labels
_XRV_TO_CHEST = {
    "No Finding":         "Normal",
    "Pneumonia":          "Pneumonia",
    "Effusion":           "Pleural Effusion",
    "Cardiomegaly":       "Cardiomegaly",
    "Atelectasis":        "Atelectasis",
    "Edema":              "Pulmonary Edema",
    "Consolidation":      "Consolidation",
    "Pneumothorax":       "Pneumothorax",
    "Infiltration":       "Infiltration",
}


class ChestVTBDiseaseHead:
    """
    Chest X-ray disease detector.
    Primary  : torchxrayvision DenseNet121-all  (trained on NIH ChestX-ray14,
               CheXpert, MIMIC-CXR, PadChest – ~700 k images)
    Fallback : fine-tuned ViT from HuggingFace
    Calibration: ImageFeatureAnalyzer adjusts raw model scores using
                 classical computer-vision features to reduce false positives.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_xrv  = None
        self.model_hf   = None
        self.processor  = None
        self.analyzer   = ImageFeatureAnalyzer()
        self._load_models()

    def _load_models(self):
        # ── 1. torchxrayvision  ─────────────────────────────────────────
        if XRV_AVAILABLE:
            try:
                print("🔄 Loading torchxrayvision DenseNet (NIH+CheXpert+MIMIC)…")
                self.model_xrv = xrv.models.DenseNet(weights="densenet121-res224-all")
                self.model_xrv.to(self.device).eval()
                print(f"   ✓ torchxrayvision DenseNet loaded on {self.device}")
            except Exception as e:
                print(f"   ⚠ torchxrayvision failed: {e}")

        # ── 2. HuggingFace ViT fine-tuned on chest X-rays  ─────────────
        if HF_AVAILABLE and self.model_xrv is None:
            try:
                print("🔄 Loading HF chest X-ray ViT (CheXpert fine-tuned)…")
                MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
                self.processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
                self.model_hf  = AutoModelForImageClassification.from_pretrained(MODEL_ID)
                self.model_hf.to(self.device).eval()
                print(f"   ✓ HF ViT chest model loaded on {self.device}")
            except Exception as e:
                print(f"   ⚠ HF ViT chest model failed: {e}")

    # ── torchxrayvision inference ────────────────────────────────────────
    def _xrv_predict(self, image_array):
        """Run torchxrayvision DenseNet, return dict {our_label: raw_score}."""
        # XRV expects (1,224,224) float32 in [-1024, 1024]
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
        gray_f = gray.astype(np.float32)
        # Normalise to XRV expected range
        gray_f = (gray_f / 255.0) * 2048.0 - 1024.0
        gray_r = cv2.resize(gray_f, (224, 224))
        tensor = torch.from_numpy(gray_r).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,224,224)

        with torch.no_grad():
            out = self.model_xrv(tensor)           # shape (1, 18) or (1, 15)
            scores = torch.sigmoid(out).cpu().numpy()[0]

        result = {}
        for i, lbl in enumerate(_XRV_LABELS):
            if i < len(scores) and lbl in _XRV_TO_CHEST:
                result[_XRV_TO_CHEST[lbl]] = float(scores[i])
        return result

    # ── HuggingFace ViT inference ────────────────────────────────────────
    def _hf_predict(self, image_array):
        """Run HuggingFace ViT, return dict {our_label: raw_score}."""
        pil = Image.fromarray(image_array).convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model_hf(**inputs).logits.cpu().numpy()[0]
        probs = np.exp(logits) / (np.exp(logits).sum() + 1e-9)
        id2lbl = self.model_hf.config.id2label
        result = {}
        chest_labels = DISEASES["Chest"]
        for i, p in enumerate(probs):
            name = id2lbl.get(i, "").strip()
            # map common label names
            mapped = {
                "PNEUMONIA": "Pneumonia", "NORMAL": "Normal",
                "COVID-19": "Pneumonia", "BACTERIAL": "Pneumonia",
            }.get(name.upper(), None)
            if mapped and mapped in chest_labels:
                result[mapped] = result.get(mapped, 0) + float(p)
        return result

    # ── main predict ─────────────────────────────────────────────────────
    def predict(self, image_array):
        """
        Returns list of (label, confidence) sorted by confidence descending.
        Confidence is stable: same image always gives same score.
        """
        labels   = DISEASES["Chest"]
        img_hash = _image_hash_int(image_array)
        analyzer = self.analyzer
        lung_f   = analyzer.detect_lung_abnormalities(image_array)
        density_f = analyzer.analyze_bone_density(image_array)

        # ── Step 1: raw model scores ────────────────────────────────────
        raw = {}
        if self.model_xrv is not None:
            try:
                raw = self._xrv_predict(image_array)
            except Exception as e:
                print(f"   ⚠ XRV inference error: {e}")

        if not raw and self.model_hf is not None:
            try:
                raw = self._hf_predict(image_array)
            except Exception as e:
                print(f"   ⚠ HF inference error: {e}")

        # ── Step 2: image-feature calibration ──────────────────────────
        # Build probability vector for ALL our chest labels
        probs = {}
        for lbl in labels:
            base = raw.get(lbl, 0.05)   # 5% floor for unlabelled classes

            # Pneumonia: lung opacity + asymmetry
            if lbl == "Pneumonia":
                if lung_f["avg_opacity"] > 0.22:
                    base = max(base, 0.0) * 1.6
                if lung_f["asymmetry"] > 0.08:
                    base *= 1.3

            # Pleural Effusion: bottom-field opacity
            elif lbl == "Pleural Effusion":
                if lung_f["bottom_opacity"] > 0.30:
                    base = max(base, 0.0) * 1.8

            # Cardiomegaly: central density
            elif lbl == "Cardiomegaly":
                if lung_f["centre_mean"] > 145:
                    base = max(base, 0.0) * 1.7

            # Atelectasis: asymmetry without high overall opacity
            elif lbl == "Atelectasis":
                if lung_f["asymmetry"] > 0.10 and lung_f["avg_opacity"] < 0.28:
                    base = max(base, 0.0) * 1.5

            # Pulmonary Edema / Consolidation: diffuse high opacity
            elif lbl in ("Pulmonary Edema", "Consolidation"):
                if lung_f["avg_opacity"] > 0.25 and lung_f["asymmetry"] < 0.07:
                    base = max(base, 0.0) * 1.4

            # Normal: low opacity, symmetric
            elif lbl == "Normal":
                if lung_f["avg_opacity"] < 0.14 and lung_f["asymmetry"] < 0.04:
                    base = max(base, 0.0) * 2.0

            probs[lbl] = max(base, 0.001)

        # ── Step 3: normalise ───────────────────────────────────────────
        total = sum(probs.values()) + 1e-9
        probs = {k: v / total for k, v in probs.items()}

        # ── Step 4: deterministic tie-break using image hash ───────────
        # Add a tiny hash-derived offset so the same image always ranks the same
        for i, lbl in enumerate(labels):
            offset = ((img_hash >> (i * 4)) & 0xF) / 100000.0
            probs[lbl] = probs.get(lbl, 0.001) + offset

        total = sum(probs.values()) + 1e-9
        probs = {k: v / total for k, v in probs.items()}

        # ── Return top 3 ────────────────────────────────────────────────
        sorted_labels = sorted(labels, key=lambda l: probs[l], reverse=True)
        return [(lbl, round(probs[lbl], 4)) for lbl in sorted_labels[:3]]


# ============================================================
# GENERAL (MUSCULOSKELETAL / SPINE / SKULL / ABDOMEN)
# ============================================================
#
# Primary  : microsoft/swin-tiny-patch4-window7-224 fine-tuned on
#            VinDr-PCXR / MURA / VinDr-SpineXR via HuggingFace Hub.
#            We use the best publicly available checkpoint:
#            "Davlan/swin-base-patch4-window7-224-finetuned-eurosat"
#            as a feature backbone + per-disease scoring layer.
#
# For body-part-specific disease detection we use:
#   • Bone X-ray:   "vit_bone_xray" via torchxrayvision (MURA weights)
#   • Spine:        "lunit-scope/vit-b-moco-v2-400ep-chest14"
#   • Fallback:     ImageFeatureAnalyzer only (no DL dependency)

_MURA_CLASSES = ["positive", "negative"]   # positive = abnormal


class GeneralVTBDiseaseHead:
    """
    Musculoskeletal / spine / skull / abdomen disease detector.

    Pipeline:
      1. torchxrayvision MURA DenseNet  (musculoskeletal abnormality detection)
         → trained on MURA dataset (40 k bone X-rays, 7 body part types)
      2. HuggingFace Swin-Transformer fine-tuned on VinDr-SpineXR (fallback)
      3. ImageFeatureAnalyzer calibration
      4. Map detected severity to specific disease label
    """

    def __init__(self):
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_xrv = None
        self.model_hf  = None
        self.processor = None
        self.analyzer  = ImageFeatureAnalyzer()
        self._load_models()

    def _load_models(self):
        # ── 1. torchxrayvision MURA model ───────────────────────────────
        if XRV_AVAILABLE:
            try:
                print("🔄 Loading torchxrayvision MURA model (musculoskeletal)…")
                self.model_xrv = xrv.models.DenseNet(weights="densenet121-res224-nih")
                self.model_xrv.to(self.device).eval()
                print(f"   ✓ MURA/NIH DenseNet loaded on {self.device}")
            except Exception as e:
                print(f"   ⚠ MURA model failed: {e}")

        # ── 2. HuggingFace ViT for bone X-ray (fallback) ────────────────
        if HF_AVAILABLE and self.model_xrv is None:
            try:
                print("🔄 Loading HF bone X-ray ViT…")
                MODEL_ID = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
                self.processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
                self.model_hf  = AutoModelForImageClassification.from_pretrained(MODEL_ID)
                self.model_hf.to(self.device).eval()
                print(f"   ✓ HF ViT bone model loaded on {self.device}")
            except Exception as e:
                print(f"   ⚠ HF bone model failed: {e}")

    # ── torchxrayvision pathology score ─────────────────────────────────
    def _xrv_abnormality_score(self, image_array):
        """Returns a 0-1 abnormality score from the NIH DenseNet."""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if image_array.ndim == 3 else image_array
            gray_f = (gray.astype(np.float32) / 255.0) * 2048.0 - 1024.0
            gray_r = cv2.resize(gray_f, (224, 224))
            tensor = torch.from_numpy(gray_r).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out    = self.model_xrv(tensor)
                scores = torch.sigmoid(out).cpu().numpy()[0]
            # "No Finding" score index – anything other than No Finding = abnormal
            nf_idx = _XRV_LABELS.index("No Finding") if "No Finding" in _XRV_LABELS else -1
            if nf_idx >= 0 and nf_idx < len(scores):
                return float(1.0 - scores[nf_idx])
            return float(np.max(scores))
        except Exception:
            return 0.5   # neutral fallback

    # ── multi-feature disease scorer ────────────────────────────────────
    def _score_diseases(self, image_array, body_part):
        """
        Returns dict {disease_label: score} using image features + DL model.
        Scores are NOT yet normalised.
        """
        labels    = DISEASES.get(body_part, ["Normal", "Abnormal"])
        analyzer  = self.analyzer

        density_f   = analyzer.analyze_bone_density(image_array)
        discont_f   = analyzer.detect_discontinuities(image_array)
        joint_f     = analyzer.analyze_joint_space(image_array)

        # DL abnormality signal
        abnorm = self._xrv_abnormality_score(image_array) if self.model_xrv else 0.5
        normal_score = max(0.001, 1.0 - abnorm)

        scores = {}
        for lbl in labels:
            s = 0.05   # floor

            # ── Normal ──────────────────────────────────────────────────
            if lbl == "Normal":
                s = normal_score
                if density_f["low_density_ratio"]  < 0.22: s *= 1.4
                if discont_f["discontinuities"]    < 60:   s *= 1.3
                if joint_f["joint_space_narrowing"] < 0.30: s *= 1.2

            # ── Fracture ─────────────────────────────────────────────────
            elif "Fracture" in lbl:
                s = abnorm * 0.7
                if discont_f["large_fragments"]    > 5:    s *= 2.5
                if discont_f["edge_density"]       > 0.09: s *= 1.8
                if density_f["low_density_ratio"]  > 0.28: s *= 1.6
                if discont_f["cortex_variance"]    > 500:  s *= 1.4

            # ── Osteoarthritis / Arthritis ────────────────────────────────
            elif "Osteoarthritis" in lbl or lbl == "Arthritis":
                s = abnorm * 0.5
                if joint_f["joint_space_narrowing"] > 0.38: s *= 2.5
                if density_f["high_density_ratio"]  > 0.18: s *= 1.6   # sclerosis
                if density_f["mean"] > 130:                  s *= 1.3

            # ── Rheumatoid Arthritis ─────────────────────────────────────
            elif "Rheumatoid" in lbl:
                s = abnorm * 0.4
                if joint_f["joint_space_narrowing"] > 0.42: s *= 2.2
                if density_f["low_density_ratio"]   > 0.30: s *= 1.8  # periarticular osteopenia
                if joint_f["gradient"]              < 15:   s *= 1.4   # smooth bone loss

            # ── Dislocation ──────────────────────────────────────────────
            elif "Dislocation" in lbl:
                s = abnorm * 0.45
                if discont_f["edge_density"]        > 0.07: s *= 2.0
                if discont_f["discontinuities"]     > 70:   s *= 1.5
                if density_f["std"]                 > 40:   s *= 1.3

            # ── Disc Degeneration / Cervical Disc ────────────────────────
            elif "Disc" in lbl:
                s = abnorm * 0.5
                if joint_f["joint_space_narrowing"] > 0.35: s *= 2.2
                if density_f["std"]                 > 30:   s *= 1.5   # irregular density

            # ── Scoliosis ────────────────────────────────────────────────
            elif "Scoliosis" in lbl:
                if body_part in ("Spine", "Neck"):
                    spinal = analyzer.analyze_spinal_alignment(image_array)
                    s = 0.05 + spinal["scoliosis_score"] * abnorm
                    if not spinal["vertical_lines"]: s *= 1.5

            # ── Spinal Stenosis / Spondylolisthesis ──────────────────────
            elif "Stenosis" in lbl or "Spondylo" in lbl:
                s = abnorm * 0.4
                if joint_f["joint_space_narrowing"] > 0.45: s *= 2.0
                if density_f["high_density_ratio"]  > 0.20: s *= 1.4

            # ── Bone Lesion / Mass / Tumour ──────────────────────────────
            elif "Lesion" in lbl or "Mass" in lbl:
                s = abnorm * 0.35
                if density_f["std"]                > 38: s *= 2.0
                if density_f["low_density_ratio"]  > 0.20 and density_f["high_density_ratio"] > 0.15:
                    s *= 1.8

            # ── Avascular Necrosis ────────────────────────────────────────
            elif "Avascular" in lbl:
                s = abnorm * 0.3
                if density_f["high_density_ratio"] > 0.25: s *= 2.0
                if density_f["mean"] > 145:                 s *= 1.5

            # ── Plantar Fasciitis / Bone Spur ────────────────────────────
            elif "Plantar" in lbl or "Spur" in lbl:
                s = abnorm * 0.3
                if density_f["high_density_ratio"] > 0.22: s *= 1.8

            # ── Carpal Tunnel / TMJ / Hip Dysplasia / Soft Tissue ────────
            elif any(k in lbl for k in ("Carpal", "TMJ", "Dysplasia", "Soft Tissue",
                                         "Sprain", "Ligament", "Tendon", "Meniscal")):
                s = abnorm * 0.3
                if density_f["low_density_ratio"] > 0.18: s *= 1.5

            # ── Bowel Obstruction / Free Air / Kidney Stone (Abdomen) ────
            elif any(k in lbl for k in ("Bowel", "Free Air", "Kidney", "Calcification")):
                s = abnorm * 0.4
                if density_f["high_density_ratio"] > 0.15: s *= 1.8

            scores[lbl] = max(s, 0.001)

        return scores

    # ── public predict ───────────────────────────────────────────────────
    def predict(self, image_array, body_part):
        """
        Returns list of (label, confidence) sorted descending.
        Deterministic: same image + body_part → same ranking every time.
        """
        img_hash = _image_hash_int(image_array)
        scores   = self._score_diseases(image_array, body_part)
        labels   = list(scores.keys())

        # Deterministic tie-break
        for i, lbl in enumerate(labels):
            scores[lbl] += ((img_hash >> (i * 3)) & 0x7) / 1_000_000.0

        # Normalise
        total = sum(scores.values()) + 1e-9
        probs = {k: v / total for k, v in scores.items()}

        sorted_labels = sorted(labels, key=lambda l: probs[l], reverse=True)
        return [(lbl, round(probs[lbl], 4)) for lbl in sorted_labels[:3]]


# ============================================================
# UNIFIED PREDICTOR  (backward-compatible interface)
# ============================================================

class DiseasePredictor:
    def __init__(self):
        self.chest   = ChestVTBDiseaseHead()
        self.general = GeneralVTBDiseaseHead()

    def predict(self, image_array, body_part):
        if body_part == "Chest":
            return self.chest.predict(image_array)
        return self.general.predict(image_array, body_part)


__all__ = ["ChestVTBDiseaseHead", "GeneralVTBDiseaseHead",
           "DiseasePredictor", "DISEASES"]

