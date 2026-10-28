"""
preprocessing.py
----------------
Functions:
  load_image()          - load H&E tif + binary/instance masks
  create_tissue_mask()  - detect tissue vs background via LAB thresholding
  macenko_normalize()   - Macenko H&E stain normalization (pure NumPy)
  tile_image()          - extract 256x256 tiles with configurable stride
  stitch_predictions()  - average-blend overlapping tile predictions back to full image
"""

import numpy as np
import cv2
from PIL import Image
import tifffile
from pathlib import Path
from skimage.draw import polygon as skimage_polygon


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """Load an H&E image (tif/png/jpg). Returns uint8 RGB array (H, W, 3)."""
    path = Path(image_path)
    try:
        # First try PIL (handles standard TIFF and LZW compression cleanly)
        img = np.array(Image.open(str(path)).convert("RGB"))
    except Exception:
        # Fallback to OpenCV
        bgr = cv2.imread(str(path))
        if bgr is not None:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to tifffile
            img = tifffile.imread(str(path))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[0] in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 4:
                img = img[:, :, :3]
    return img.astype(np.uint8)


def load_mask(mask_path: str) -> np.ndarray:
    """Load a binary or instance mask. Returns uint32 array (H, W)."""
    path = Path(mask_path)
    try:
        mask = np.array(Image.open(str(path)))
    except Exception:
        mask = tifffile.imread(str(path))
    return mask.astype(np.uint32)


# ---------------------------------------------------------------------------
# XML → mask conversion
# ---------------------------------------------------------------------------

def xml_to_masks(xml_path: str, image_shape: tuple) -> tuple:
    """
    Convert MoNuSeg XML annotation to binary and instance masks.

    Args:
        xml_path: path to the XML annotation file
        image_shape: (H, W) of the corresponding image

    Returns:
        binary_mask   : np.uint8  (H, W), values in {0, 1}
        instance_mask : np.uint32 (H, W), nucleus i has label i (1-indexed)
    """
    import xml.etree.ElementTree as ET

    H, W = image_shape
    binary_mask = np.zeros((H, W), dtype=np.uint8)
    instance_mask = np.zeros((H, W), dtype=np.uint32)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    nucleus_id = 1
    # MoNuSeg XML structure: Annotation > Regions > Region > Vertices > Vertex
    for region in root.iter("Region"):
        vertices = region.findall(".//Vertex")
        if not vertices:
            continue
        xs = np.array([float(v.attrib["X"]) for v in vertices])
        ys = np.array([float(v.attrib["Y"]) for v in vertices])

        # Clip to image bounds
        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        rr, cc = skimage_polygon(ys, xs, shape=(H, W))
        binary_mask[rr, cc] = 1
        instance_mask[rr, cc] = nucleus_id
        nucleus_id += 1

    return binary_mask, instance_mask


# ---------------------------------------------------------------------------
# Tissue masking
# ---------------------------------------------------------------------------

def create_tissue_mask(image: np.ndarray, threshold: int = 200) -> np.ndarray:
    """
    Detect tissue vs. white background using LAB color space.

    Returns binary mask: 1 = tissue, 0 = background.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]

    # Tissue is darker than white background
    tissue = (L < threshold).astype(np.uint8)

    # Morphological cleanup to remove small holes and specks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_OPEN, kernel)

    return tissue


# ---------------------------------------------------------------------------
# Macenko stain normalization (pure NumPy)
# ---------------------------------------------------------------------------

def _rgb_to_od(image: np.ndarray, beta: float = 0.15) -> np.ndarray:
    """Convert RGB image to optical density. Clip near-zero values."""
    image = image.astype(np.float64)
    image = np.clip(image, 1, 255)
    od = -np.log(image / 255.0)
    # Remove near-transparent pixels
    od_flat = od.reshape(-1, 3)
    mask = np.all(od_flat > beta, axis=1)
    return od, od_flat, mask


def _get_stain_matrix(od_flat: np.ndarray, mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Estimate 2-stain matrix via SVD on OD values of tissue pixels.
    Returns stain_matrix of shape (2, 3): rows are H and E stain vectors.
    """
    od_tissue = od_flat[mask]
    if od_tissue.shape[0] < 10:
        # Fallback to standard H&E stain matrix
        return np.array([[0.644, 0.717, 0.267],
                         [0.093, 0.954, 0.283]])

    _, _, Vt = np.linalg.svd(od_tissue, full_matrices=False)
    # Project onto the plane of first 2 singular vectors
    plane = Vt[:2].T  # (3, 2)
    proj = od_tissue @ plane  # (N, 2)

    # Find extreme vectors (robust percentile)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    phi1 = np.percentile(angles, alpha)
    phi2 = np.percentile(angles, 100 - alpha)

    v1 = plane @ np.array([np.cos(phi1), np.sin(phi1)])
    v2 = plane @ np.array([np.cos(phi2), np.sin(phi2)])

    # Ensure positive components (flip if needed)
    if v1[0] < 0: v1 = -v1
    if v2[0] < 0: v2 = -v2

    # Assign: hematoxylin typically has higher blue component (index 2)
    if v1[2] > v2[2]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])

    # Normalize rows
    HE = HE / np.linalg.norm(HE, axis=1, keepdims=True)
    return HE


def macenko_normalize(
    image: np.ndarray,
    target_concentrations: tuple = (None, None),
    alpha: float = 1.0,
    beta: float = 0.15,
) -> np.ndarray:
    """
    Macenko stain normalization.

    Fit on the whole image (not on tiles). Normalize to target slide statistics.
    If target_concentrations is (None, None), returns the source stain matrix
    and max concentrations so you can use them as the reference.

    Args:
        image: uint8 RGB (H, W, 3)
        target_concentrations: (maxC_ref shape (2,), HE_ref shape (2,3)) from a reference slide
        alpha, beta: algorithm parameters

    Returns:
        normalized uint8 RGB (H, W, 3)
    """
    H, W = image.shape[:2]
    od, od_flat, mask = _rgb_to_od(image, beta=beta)
    HE = _get_stain_matrix(od_flat, mask, alpha=alpha)

    # Solve for concentrations: od = C @ HE  →  C = od / HE (NNLS per pixel)
    # Vectorized approximate solve via pseudo-inverse
    HE_pinv = np.linalg.pinv(HE)  # (3, 2)
    C = od_flat @ HE_pinv          # (N_pixels, 2)
    C = np.clip(C, 0, None)

    maxC = np.percentile(C, 99, axis=0)  # (2,)

    if target_concentrations[0] is None:
        # Caller wants to use this image as reference
        return HE, maxC

    maxC_ref, HE_ref = target_concentrations
    # Normalize concentrations
    C_norm = C * (maxC_ref / (maxC + 1e-8))

    # Reconstruct OD then RGB
    od_norm = C_norm @ HE_ref        # (N_pixels, 3)
    od_norm = od_norm.reshape(H, W, 3)
    img_norm = np.exp(-od_norm) * 255.0
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    return img_norm


def fit_macenko_reference(image: np.ndarray, alpha: float = 1.0, beta: float = 0.15):
    """
    Fit Macenko parameters on a reference image.
    Returns (maxC, HE) to pass into macenko_normalize().
    """
    HE, maxC = macenko_normalize(image, target_concentrations=(None, None), alpha=alpha, beta=beta)
    return maxC, HE


# ---------------------------------------------------------------------------
# Tiling and stitching
# ---------------------------------------------------------------------------

def tile_image(
    image: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
) -> list:
    """
    Extract overlapping tiles from an image.

    Args:
        image: (H, W) or (H, W, C) numpy array
        tile_size: square tile side length
        stride: step between tile origins

    Returns:
        List of dicts: [{tile, row, col, h_start, w_start}]
    """
    if image.ndim == 2:
        H, W = image.shape
    else:
        H, W = image.shape[:2]

    tiles = []
    row = 0
    r = 0
    while r < H:
        r_end = min(r + tile_size, H)
        r_start = max(r_end - tile_size, 0)

        c = 0
        col = 0
        while c < W:
            c_end = min(c + tile_size, W)
            c_start = max(c_end - tile_size, 0)

            if image.ndim == 2:
                tile = image[r_start:r_end, c_start:c_end]
                # Pad if smaller than tile_size
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad = np.zeros((tile_size, tile_size), dtype=image.dtype)
                    pad[:tile.shape[0], :tile.shape[1]] = tile
                    tile = pad
            else:
                tile = image[r_start:r_end, c_start:c_end, :]
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                    pad[:tile.shape[0], :tile.shape[1], :] = tile
                    tile = pad

            tiles.append({
                "tile": tile,
                "row": row,
                "col": col,
                "h_start": r_start,
                "w_start": c_start,
            })

            if c_end == W:
                break
            c += stride
            col += 1

        if r_end == H:
            break
        r += stride
        row += 1

    return tiles


def stitch_predictions(
    tiles: list,
    image_shape: tuple = None,
    output_shape: tuple = None,
    tile_size: int = 256,
) -> np.ndarray:
    """
    Average-blend overlapping tile predictions back to full image.

    Args:
        tiles: list of dicts with keys 'pred' or 'tile' (H_t, W_t) float, 'h_start', 'w_start'
        image_shape / output_shape: (H, W) of original image
        tile_size: square tile side length

    Returns:
        (H, W) float array with averaged predictions
    """
    shape = output_shape if output_shape is not None else image_shape
    if shape is None:
        raise ValueError("Must provide image_shape or output_shape")
    H, W = shape
    accumulator = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)

    for t in tiles:
        pred = t.get("pred", t.get("tile"))   # (tile_size, tile_size) or smaller at boundary
        rs = t["h_start"]
        cs = t["w_start"]

        r_end = min(rs + tile_size, H)
        c_end = min(cs + tile_size, W)
        ph = r_end - rs
        pw = c_end - cs

        accumulator[rs:r_end, cs:c_end] += pred[:ph, :pw]
        weight[rs:r_end, cs:c_end] += 1.0

    weight = np.where(weight == 0, 1, weight)
    return accumulator / weight
