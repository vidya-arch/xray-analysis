


import numpy as np
import cv2
import torch
from PIL import Image
from scipy import ndimage


class HeatmapGenerator:
    """
    Generates attention heatmaps using CLIP's visual transformer.
    Shows which regions of the X-ray the AI focuses on.
    Enhanced with bounding box detection.
    """

    def __init__(self, clip_detector):
        self.model = clip_detector.model
        self.preprocess = clip_detector.preprocess
        self.tokenizer = clip_detector.tokenizer
        self.device = clip_detector.device

    # ── public API ───────────────────────────────────────────────────────

    def generate(self, image_array, body_part, diseases):
        """
        Generate a heatmap overlay showing disease-relevant regions.

        Parameters
        ----------
        image_array : numpy array (H, W) or (H, W, 3)
        body_part   : str — detected body part
        diseases    : list of (name, prob) tuples

        Returns
        -------
        dict with keys:
            'heatmap': numpy RGB array of the overlaid heatmap
            'bboxes': list of bounding boxes [(x, y, w, h, confidence), ...]
            'raw_heatmap': normalized heatmap array (0-1)
        """
        if self.model is None:
            return self._fallback_heatmap(image_array, body_part, diseases)
        try:
            return self._clip_attention_heatmap(image_array, body_part, diseases)
        except Exception as e:
            print(f"⚠ CLIP heatmap error: {e} — using gradient fallback")
            return self._fallback_heatmap(image_array, body_part, diseases)

    # ── CLIP patch-level similarity heatmap ──────────────────────────────

    def _clip_attention_heatmap(self, image_array, body_part, diseases):
        # Convert to PIL
        if len(image_array.shape) == 3:
            image_pil = Image.fromarray(image_array)
        else:
            image_pil = Image.fromarray(image_array).convert("RGB")

        image_input = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        # Build text prompt for the primary finding
        primary_disease = diseases[0][0] if diseases else "abnormality"
        if primary_disease == "Normal":
            text_prompt = f"a medical X-ray of a normal healthy {body_part}"
        else:
            text_prompt = f"a medical X-ray showing {primary_disease} in the {body_part}"

        text_tokens = self.tokenizer([text_prompt]).to(self.device)

        # ── Hook into last transformer block ────────────────────────────
        features_store = {}

        def hook_fn(_module, _input, output):
            features_store["output"] = output

        visual = self.model.visual
        resblocks = visual.transformer.resblocks
        hook = resblocks[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            _ = self.model.encode_image(image_input)

        hook.remove()

        # ── Extract patch features ──────────────────────────────────────
        output = features_store["output"]

        # Handle both (batch, seq, dim) and (seq, batch, dim)
        if output.shape[0] == 1:
            patch_tokens = output[0, 1:, :]        # skip CLS
        else:
            patch_tokens = output[1:, 0, :]        # skip CLS

        # Apply post-layer-norm + projection (same transform CLIP uses)
        patch_tokens = visual.ln_post(patch_tokens)
        if hasattr(visual, "proj") and visual.proj is not None:
            patch_tokens = patch_tokens @ visual.proj

        patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

        # Per-patch similarity with disease text
        similarity = (patch_tokens @ text_features.T).squeeze(-1).detach().cpu().numpy()

        # Reshape to spatial grid (7×7 for ViT-B/32 @ 224 px)
        num_patches = similarity.shape[0]
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            grid_size = int(np.ceil(np.sqrt(num_patches)))
            similarity = np.pad(similarity, (0, grid_size * grid_size - num_patches))

        heatmap = similarity.reshape(grid_size, grid_size)
        heatmap = self._normalise(heatmap)

        # Up-sample to original image size
        h, w = image_array.shape[:2]
        heatmap = cv2.resize(heatmap.astype(np.float32), (w, h),
                             interpolation=cv2.INTER_CUBIC)

        # Gaussian smooth for nicer visualisation
        k = self._odd(max(w, h) // 8)
        heatmap = cv2.GaussianBlur(heatmap, (k, k), 0)
        heatmap = self._normalise(heatmap)

        # Detect bounding boxes from heatmap
        bboxes = self._detect_bounding_boxes(heatmap, primary_disease)

        print(f"🔥 Heatmap generated — focus: {primary_disease} ({body_part})")
        print(f"📦 Detected {len(bboxes)} abnormal regions")

        overlay = self._apply_overlay(image_array, heatmap, body_part, primary_disease, bboxes)

        return {
            'heatmap': overlay,
            'bboxes': bboxes,
            'raw_heatmap': heatmap
        }

    # ── Bounding box detection from heatmap ──────────────────────────────

    def _detect_bounding_boxes(self, heatmap, disease_name):
        """
        Detect bounding boxes around high-attention regions in the heatmap.

        Parameters
        ----------
        heatmap : numpy array (H, W) normalized to 0-1
        disease_name : str

        Returns
        -------
        List of tuples: [(x, y, w, h, confidence), ...]
        """
        # Threshold the heatmap to find high-attention regions
        threshold = 0.6  # Adjust based on your needs
        binary_map = (heatmap > threshold).astype(np.uint8)

        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_map, connectivity=8
        )

        bboxes = []
        min_area = heatmap.size * 0.01  # Minimum 1% of image area

        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Filter small regions
            if area < min_area:
                continue

            # Calculate confidence based on average heatmap value in this region
            region_mask = (labels == i)
            confidence = float(np.mean(heatmap[region_mask]))

            # Add some padding
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(heatmap.shape[1] - x, w + 2 * pad)
            h = min(heatmap.shape[0] - y, h + 2 * pad)

            bboxes.append((x, y, w, h, confidence))

        # Sort by confidence (highest first)
        bboxes.sort(key=lambda b: b[4], reverse=True)

        # Return top 3 regions
        return bboxes[:3]

    # ── gradient-based fallback ──────────────────────────────────────────

    def _fallback_heatmap(self, image_array, body_part, diseases):
        gray = (cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                if len(image_array.shape) == 3 else image_array.copy())
        gray = gray.astype(np.float32)

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag = self._normalise(mag).astype(np.float32)

        h, w = gray.shape[:2]
        k = self._odd(max(w, h) // 6)
        mag = cv2.GaussianBlur(mag, (k, k), 0)
        mag = self._normalise(mag)

        bboxes = self._detect_bounding_boxes(mag, diseases[0][0] if diseases else "Analysis")
        primary = diseases[0][0] if diseases else "Analysis"
        overlay = self._apply_overlay(image_array, mag, body_part, primary, bboxes)

        return {
            'heatmap': overlay,
            'bboxes': bboxes,
            'raw_heatmap': mag
        }

    # ── overlay rendering ────────────────────────────────────────────────

    def _apply_overlay(self, image_array, heatmap, body_part, disease, bboxes):
        h, w = image_array.shape[:2]

        # Colormap
        heatmap_u8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Prepare base image
        if len(image_array.shape) == 2:
            img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image_array.copy()
        if img_rgb.dtype != np.uint8:
            img_rgb = ((img_rgb * 255) if img_rgb.max() <= 1.0
                       else img_rgb).astype(np.uint8)

        # Blend
        overlay = cv2.addWeighted(img_rgb, 0.55, heatmap_color, 0.45, 0)

        # Draw bounding boxes
        for idx, (x, y, bbox_w, bbox_h, conf) in enumerate(bboxes):
            # Color based on confidence (red for high, yellow for medium)
            if conf > 0.8:
                color = (255, 0, 0)  # Red
            elif conf > 0.65:
                color = (255, 165, 0)  # Orange
            else:
                color = (255, 255, 0)  # Yellow

            # Draw rectangle
            thickness = max(2, int(min(w, h) / 300))
            cv2.rectangle(overlay, (x, y), (x + bbox_w, y + bbox_h), color, thickness)

            # Add label
            label = f"#{idx+1}: {conf*100:.0f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.35, min(w, h) / 800)
            th = max(1, int(fs * 2))
            (tw, text_h), baseline = cv2.getTextSize(label, font, fs, th)

            # Label background
            label_y = max(y - 5, text_h + 5)
            cv2.rectangle(overlay,
                         (x, label_y - text_h - 5),
                         (x + tw + 8, label_y + baseline),
                         color, -1)
            cv2.putText(overlay, label, (x + 4, label_y - 3),
                       font, fs, (0, 0, 0), th, cv2.LINE_AA)

        # ── Main label ────────────────────────────────────────────────────
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"AI Focus: {disease} ({body_part})"
        fs = max(0.40, min(w, h) / 640)
        th = max(1, int(fs * 2))
        (tw, text_h), _ = cv2.getTextSize(label, font, fs, th)
        pad = 8
        cv2.rectangle(overlay,
                      (pad, h - text_h - pad * 3),
                      (tw + pad * 3, h - pad),
                      (0, 0, 0), -1)
        cv2.putText(overlay, label,
                    (pad * 2, h - pad * 2),
                    font, fs, (255, 255, 255), th, cv2.LINE_AA)

        # ── Colour-bar legend on the right edge ─────────────────────────
        bar_w = max(14, w // 28)
        bar_h = h // 3
        bar_x = w - bar_w - 14
        bar_y = (h - bar_h) // 2

        for i in range(bar_h):
            frac = 1.0 - i / bar_h
            c_val = int(frac * 255)
            row_bgr = cv2.applyColorMap(
                np.array([[c_val]], dtype=np.uint8), cv2.COLORMAP_JET
            )[0][0]
            row_rgb = (int(row_bgr[2]), int(row_bgr[1]), int(row_bgr[0]))
            cv2.line(overlay, (bar_x, bar_y + i),
                     (bar_x + bar_w, bar_y + i), row_rgb, 1)

        sfs = max(0.30, fs * 0.70)
        sth = max(1, int(sfs * 2))
        cv2.putText(overlay, "High", (bar_x - 4, bar_y - 6),
                    font, sfs, (255, 255, 255), sth, cv2.LINE_AA)
        cv2.putText(overlay, "Low", (bar_x, bar_y + bar_h + 14),
                    font, sfs, (255, 255, 255), sth, cv2.LINE_AA)

        return overlay

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-8)

    @staticmethod
    def _odd(n):
        n = max(n, 3)
        return n if n % 2 == 1 else n + 1

