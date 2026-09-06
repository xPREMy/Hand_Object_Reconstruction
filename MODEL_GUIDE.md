# HORT Model — Complete Beginner Guide

> **Goal**: Given a single photo of a hand holding an object → output a 3D point cloud of that object.

---

## Can I Download a Dataset and Train?

**YES** — but with limitations on your laptop.

| Dataset | Size | Can Download? | Training Feasible on RTX 4050 (6GB)? |
|---------|------|---------------|--------------------------------------|
| **ObMan** | ~18 GB | ✅ Free, public | ⚠️ Yes, but reduce batch size to 8–16 (paper uses 192 on 4× A100) |
| **HO3D v3** | ~10 GB | ✅ Free (registration required) | ⚠️ Yes |
| **DexYCB** | ~30 GB | ✅ Free, public | ⚠️ Yes |
| **MOW** | ~2 GB | ✅ Free | ✅ Small enough |

**Practical advice for your laptop:**
```bash
# ObMan is the best starting point (Phase 1)
# Download from: https://www.di.ens.fr/willow/research/obman/data/

python train.py --config config.yaml --dataset obman --epochs 50 --batch_size 8
```
The paper trained on 4× NVIDIA A100 GPUs for ~50 hours.
On your RTX 4050 with batch_size=8, expect ~10–15× longer (roughly a few weeks for full training).
For quick experiments, use `--max_samples 5000` to train on a subset.

---

## Phase Overview (What the Model Does)

```
INPUT: Single RGB photo (224×224)
         +
       Hand mesh (778 vertices, estimated from photo)
         │
         ▼
┌─────────────────┐
│  PHASE 1        │  Image Encoder + Hand Encoder
│  Feature        │  → extract visual features from photo
│  Extraction     │  → extract geometric hand features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 2        │  Sparse Transformer Decoder
│  Coarse         │  → predict WHERE the object is (translation)
│  Reconstruction │  → predict a rough 3D shape (2048 points)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 3        │  Dense Decoder
│  Fine           │  → project 2048 points onto image
│  Reconstruction │  → fetch fine-grained visual details
│                 │  → upsample to 16,384 detailed points
└────────┬────────┘
         │
         ▼
OUTPUT: Dense 3D point cloud of the hand-held object (16,384 points)
        + Object 3D translation (where it is relative to the hand)
```

---

## Phase 1A — Image Encoder

### What it is
A **DINOv2-Large** Vision Transformer (ViT) pre-trained by Meta on 142 million images using self-supervised learning. It is very good at understanding image semantics without needing labels.

### Why DINOv2?
- Pre-trained on huge data → rich visual features out of the box
- Works well for 3D-related tasks (understands depth, texture, shape)
- Better than training a visual encoder from scratch on small hand-object datasets

### Input
```
image: (Batch, 3, 224, 224)
  ↑
  Cropped around the hand-object region
  Resized to 224×224
  Normalized with ImageNet mean/std
```

### How it works (internally)
1. The image is split into **16×16 non-overlapping patches** → 256 patches
2. Each patch is embedded as a 1024-dimensional token
3. A special **[CLS] token** is prepended (represents the whole image)
4. All 257 tokens pass through **24 Transformer layers**
5. First 12 layers are **frozen** (pretrained knowledge preserved)
6. Last 12 layers are **fine-tuned** (adapted to hand-object domain)

### Output
```
fv: (Batch, 257, 1024)
     ↑       ↑     ↑
     │      tokens  feature size per token
     │
     1 CLS token + 256 patch tokens
```

---

## Phase 1B — Hand Encoder

### What it is
A **PointNet** — a neural network designed to process 3D point clouds. Here it processes the 778 vertices of the MANO hand mesh.

### Why encode the hand?
- The object position/shape depends heavily on HOW the hand is holding it
- Without hand info, the model can't tell if the object is a cup, bottle, or ball
- Using all 778 vertices gives fine-grained shape information (not just joint positions)

### Input
```
hand_verts:  (Batch, 778, 3)   ← 778 MANO hand mesh vertices in 3D camera space
hand_joints: (Batch, 21, 3)    ← 21 joint positions (16 base joints + 5 fingertips)
palm_coord:  (Batch, 3)        ← position of the palm center in 3D
```

### How it works (internally)
**Step 1: Create 22 local coordinate systems**
- The paper uses 22 "reference frames" centered at:
  - 16 hand joints (knuckles, finger segments)
  - 5 fingertips (thumb, index, middle, ring, pinky)
  - 1 palm center
- For EACH of the 778 vertices, compute its offset from EACH of the 22 centers:
  ```
  offset[vertex_i, frame_k] = vertex_i_position - center_k_position
  ```
- This gives 778 vertices × 22 frames × 3 coordinates = **778 × 66 numbers**

**Step 2: Append vertex index**
- Add a normalized vertex ID (0.0 to 1.0) for each vertex
- Final input: **778 × 67** per sample

**Why local coordinate systems?**
Because if you only use absolute positions, the model has to learn "what a hand looks like" for every camera position. Local coordinates make the representation rotation/translation invariant — it only cares about the *shape* of the hand, not where the camera is.

**Step 3: 5-layer PointNet MLP**
```
67 → 128 → 256 → 512 → 1024 → 1024
```
Applied independently to each of the 778 vertices (shared weights).

**Step 4: Global Max Pooling**
- Take the maximum across all 778 vertices at each feature dimension
- This gives a single 1024-D vector representing the entire hand shape

### Output
```
fh: (Batch, 1024)   ← hand feature vector
```

---

## Phase 2 — Sparse Transformer Decoder

### What it is
A standard **Transformer Decoder** (like in BERT/GPT but used for 3D generation).

### Why jointly predict translation AND points?
- Previous works used **separate networks** for object pose and shape
- Using one transformer lets the shape prediction inform the pose and vice versa
- More parameter-efficient and trains end-to-end

### Input
```
fv: (Batch, 257, 1024)   ← image features (the "memory" to attend to)
fh: (Batch, 1024)        ← hand feature (also added to memory)
```

### How it works (internally)

**Learnable Queries (what the model "asks"):**
```
1 pose token   (what is the object's position?)
+ 2048 point tokens (what are the object's 3D point positions?)
= 2049 total learnable tokens, each of size 512
```

**Transformer Decoder (10 layers):**
Each layer does:
1. **Self-Attention**: All 2049 queries talk to each other (pose informs points and vice versa)
2. **Cross-Attention**: Queries look at image features + hand features to gather information

**Prediction heads (simple MLPs after the transformer):**
```
Pose token   → MLP → (Batch, 3)         object translation t_o
Point tokens → MLP → (Batch, 2048, 3)   sparse point cloud p_s
```

**Note on Rotation**: The paper intentionally **does NOT predict rotation**. Many objects like bottles, cups, and balls are symmetric — predicting rotation is ambiguous (a cup rotated 90° looks the same from many angles). Translation is enough.

### Output
```
pred_trans:    (Batch, 3)        ← object center offset from palm (in meters)
sparse_points: (Batch, 2048, 3)  ← rough 3D point cloud of the object
```

---

## Phase 3 — Dense Decoder

### What it is
A **two-block progressive upsampling network** that takes 2048 coarse points and produces 16,384 fine-grained points. It uses pixel-aligned image features to add local surface detail.

### Why upsample?
- 2048 points gives a rough object shape
- 16,384 points captures surface details (edges, curves, textures mapped to geometry)
- Simply repeating the 2048 points would give blocky results
- Fetching pixel-level visual features adds detail aligned with what the image shows

### Input
```
p_s:        (Batch, 2048, 3)    ← sparse points from Phase 2
to:         (Batch, 3)          ← object translation from Phase 2
palm_coord: (Batch, 3)          ← hand palm position
fv:         (Batch, 257, 1024)  ← image tokens from Phase 1A
hand_verts: (Batch, 778, 3)     ← hand mesh vertices
cam_intr:   (Batch, 3, 3)       ← camera intrinsics matrix K
```

### How it works (internally)

**Step 1: Build a spatial image feature map**
- Remove the CLS token: `fv[:, 1:, :]` → **(Batch, 256, 1024)**
- The 256 patch tokens correspond to a **16×16 grid** of image patches
- Reshape: **(Batch, 1024, 16, 16)** — now it's a spatial map
- Apply 3×3 convolutions to reduce channels: **(Batch, 128, 16, 16)**
- This is the refined feature map `f_v^r` — a compact spatial representation of the image

**Step 2: Project 3D points onto image → fetch visual features**
```
Camera coordinates:  p_cam = p_s + palm + translation
                     (move sparse points from local to camera space)
                          ↓
Perspective projection:  u = fx * (X/Z) + cx
                          v = fy * (Y/Z) + cy
                          (3D point → 2D pixel location)
                          ↓
Bilinear sampling:   sample f_v^r at (u, v)
                          ↓
f_obj: (Batch, 2048, 128)  ← visual features per object point
```
Same thing is done for the 778 hand vertices → `f_hand: (Batch, 778, 128)`

**Step 3: Combine coordinates with visual features**
```
Object: [p_s coordinates, visual feature] → Linear → (Batch, 2048, 128)
Hand:   [vertex coordinates, visual feature] → Linear → (Batch, 778, 128)
```

**Step 4: Two progressive upsampling blocks**

*Block 1 — ×2 upsampling (2048 → 4096 points):*
1. **Local kNN self-attention (k=16)**: For each of the 2048 points, find its 16 nearest neighbors (from both object points and hand vertices). Aggregate their features using attention.
2. **Feature expand**: Expand each feature from 128 → 256, then reshape to 4096 × 128
3. **Offset MLP**: Predict a small 3D offset for each of the 4096 points
4. **Final coords**: base position (repeated parent) + offset

*Block 2 — ×4 upsampling (4096 → 16,384 points):*
Same process, but 4× expansion instead of 2×

**Why kNN self-attention?**
Each point needs to know about its neighbors to predict a smooth, consistent surface. Without it, each point would be predicted independently → noise and gaps in the surface.

### Output
```
dense_points: (Batch, 16384, 3)  ← final dense 3D object point cloud
```

---

## Loss Function (How Training Works)

Training uses **3 losses** combined:

```
Total Loss = 2.0 × Pose_Loss + 2.0 × Sparse_CD + 1.0 × Dense_CD
```

| Loss | Formula | What it penalizes |
|------|---------|-------------------|
| `Pose_Loss` | L1( predicted_translation, GT_translation ) | Wrong object position |
| `Sparse_CD` | Chamfer Distance( 2048 predicted pts, 2048 GT pts ) | Wrong coarse shape |
| `Dense_CD` | Chamfer Distance( 16384 predicted pts, 16384 GT pts ) | Wrong fine shape |

**Chamfer Distance** between two point clouds A and B:
```
CD(A, B) = mean over each point in A of [min distance to B]
         + mean over each point in B of [min distance to A]
```
It measures how well two point clouds match without needing point-to-point correspondence.

---

## Training Setup

```yaml
optimizer:   Adam (lr = 1e-4)
scheduler:   Cosine annealing (decays lr smoothly to 1e-6)
batch_size:  192 (paper) / 8-16 (your laptop)
epochs:      50
```

**During training:**  Use ground-truth hand pose + camera parameters
**During inference:** Use estimated hand pose from WiLoR/HaMeR + estimated camera

---

## Data Flow Summary (End to End)

```
                    Raw Image (e.g. 1920×1080)
                           │
                    [Crop around hand region]
                    [Resize to 224×224]
                    [Adjust camera intrinsics]
                           │
                 ┌─────────┴──────────┐
                 │                    │
         [Image Encoder]      [Hand Encoder]
         DINOv2-Large         PointNet (5 layers)
         (Batch, 257, 1024)   (Batch, 1024)
                 │                    │
                 └─────────┬──────────┘
                           │
                  [Sparse Decoder]
                  Transformer (10 layers, 8 heads)
                  2049 learnable queries
                           │
                  ┌────────┴────────┐
                  │                 │
           [Translation]    [Sparse Cloud]
           (Batch, 3)       (Batch, 2048, 3)
                  └────────┬────────┘
                           │ (+ image tokens + hand verts + camera K)
                  [Dense Decoder]
                  Project → Sample features → kNN attention → Upsample ×2 → Upsample ×4
                           │
                  [Dense Cloud]
                  (Batch, 16384, 3)
```

---

## File Map (Which Code Does What)

| File | Corresponds to |
|------|---------------|
| `models/image_encoder.py` | Phase 1A — DINOv2 image features |
| `models/hand_encoder.py` | Phase 1B — PointNet hand geometry |
| `models/sparse_decoder.py` | Phase 2 — Transformer sparse decoder |
| `models/dense_decoder.py` | Phase 3 — Dense upsampling decoder |
| `models/hort.py` | Top-level: connects all 4 phases |
| `losses/chamfer.py` | Chamfer Distance + composite loss |
| `datasets/base.py` | Image crop, resize, augmentation, dummy data |
| `datasets/obman.py` | ObMan real dataset reader |
| `train.py` | Full training loop |
| `validate.py` | Evaluation (CD, F-score metrics) |
| `infer.py` | Run on a single image → export .ply files |
| `visualize.py` | View .ply files as PNG or interactive 3D |
| `tests/test_hort.py` | Shape checks, loss tests, forward/backward tests |

