# HW04 Grading Guide
## Conditional Generative Models for Medical Image Synthesis
**Course**: CSYE 7374 — Deep Learning and Generative AI in Healthcare

---

## How to Use This File

This guide provides expected solutions, common mistakes, and partial-credit guidelines for every graded item in HW04. Each student submits `FirstName_LastName_HW04.ipynb`. Load the notebook, run it top-to-bottom, and compare against this guide.

---

## Part 1 — Data Loading (5 pts)

**cell-8 — Load val and test splits**

Expected solution:
```python
val_ds  = DataClass(split='val',  transform=tfm, download=True)
test_ds = DataClass(split='test', transform=tfm, download=True)
```

| Score | Criteria |
|-------|----------|
| 5 | Both splits loaded with correct `split=` argument and same `tfm` transform |
| 3 | One split correct, other missing or wrong split name |
| 1 | Attempted but wrong class, missing transform, or runtime error |
| 0 | Not attempted |

---

## Part 2 — Sample Visualisation (5 pts)

**cell-10 — 2 samples per class grid**

Expected: A 2-row × 4-column figure with grayscale OCT images. Row 0 column titles must match CLASS_NAMES = `['choroidal neovascularization', 'diabetic retinopathy', 'drusen', 'normal']` (or abbreviated forms). Images displayed with `cmap='gray'`.

| Score | Criteria |
|-------|----------|
| 5 | Correct 2×4 grid, all 4 class titles present, images visible |
| 3 | Grid correct but titles missing or wrong class in one column |
| 2 | Grid rendered but wrong dimensions or no titles |
| 1 | Attempted, figure shows but clearly incorrect |
| 0 | Not attempted or no output |

---

## Part 3 — cGAN: Generator.forward() (15 pts)

**cell-13**

Expected solution:
```python
def forward(self, z, c):
    emb = self.class_emb(c)                        # (B, 16)
    x   = torch.cat([z, emb], dim=1)               # (B, LATENT_DIM + 16)
    x   = self.fc(x).view(-1, 256, 7, 7)           # (B, 256, 7, 7)
    return self.decoder(x)                          # (B, N_CH, 28, 28)
```

Key checks:
- `self.class_emb(c)` called with c (not one-hot or float)
- Concatenation is along `dim=1` (not dim=0)
- Reshape uses `-1` or `x.size(0)` for batch dimension
- `.view(-1, 256, 7, 7)` — must match fc output size `256*7*7 = 12544`
- Smoke-test cell must print `Generator output shape: torch.Size([4, 1, 28, 28])`

| Score | Criteria |
|-------|----------|
| 15 | Correct; smoke-test passes |
| 12 | Correct logic but minor issue (e.g., wrong dim for cat — still runs) |
| 8  | Mostly correct but reshape wrong (e.g., missing 256 channels) |
| 5  | Class embedding used but concatenation or reshape missing/wrong |
| 0  | `pass` not removed, or fatal error |

---

## Part 4 — cGAN: Discriminator.forward() (10 pts)

**cell-14**

Expected solution:
```python
def forward(self, img, c):
    c_map = self.class_emb(c).view(-1, 1, Config.IMG_SIZE, Config.IMG_SIZE)  # (B, 1, 28, 28)
    x     = torch.cat([img, c_map], dim=1)                                    # (B, 2, 28, 28)
    return self.net(x)                                                         # (B, 1)
```

Key checks:
- `self.class_emb(c)` → shape `(B, 784)`
- Reshape to `(B, 1, 28, 28)` — must use `Config.IMG_SIZE` or literal `28`
- Concatenation along `dim=1` (channel dimension)
- Smoke-test must print `Discriminator output shape: torch.Size([4, 1])`

| Score | Criteria |
|-------|----------|
| 10 | Correct; smoke-test passes |
| 7  | Correct logic but wrong reshape (e.g., `view(-1, 784)` not reshaped to spatial) |
| 5  | Cat along wrong dim but otherwise correct |
| 2  | Attempted but logit shape wrong |
| 0  | `pass` not removed or fatal error |

---

## Part 5 — cGAN: Training Step (15 pts)

**cell-18 — `train_cgan_epoch`**

Expected D step:
```python
z         = torch.randn(B, Config.LATENT_DIM, device=device)
fake_imgs = G(z, labels)
d_real    = bce(D(imgs,               labels), real)
d_fake    = bce(D(fake_imgs.detach(), labels), fake)
loss_D    = 0.5 * (d_real + d_fake)
opt_D.zero_grad(); loss_D.backward(); opt_D.step()
```

Expected G step:
```python
z        = torch.randn(B, Config.LATENT_DIM, device=device)
gen_imgs = G(z, labels)
loss_G   = bce(D(gen_imgs, labels), real)
opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

Key checks:
- `.detach()` on fake images for D step (prevents gradients flowing to G)
- Fresh `z` sampled for G step (not reusing D-step z)
- `real` target used for G loss (generator wants D to say "real")
- `zero_grad()` called before `backward()` for each optimizer
- Training output is printed every 10 epochs

Common mistakes:
- Missing `.detach()` on fake_imgs in D step (no points deducted if training still works, but note it)
- Using same `z` for both D and G step (minor, note but don't deduct if both losses are non-None)
- Forgetting `zero_grad()` for one of the optimizers (deduct 3 pts)

| Score | Criteria |
|-------|----------|
| 15 | Both steps correct; training loss printed |
| 11 | Both steps correct but missing `.detach()` in D step |
| 8  | One step correct, other has bug |
| 5  | Correct structure but gradient flow issues (e.g. no zero_grad) |
| 2  | Training loop runs but loss_D or loss_G remains None |
| 0  | Not attempted or fatal error |

---

## Part 6 — DDPM: q_sample (10 pts)

**cell-24**

Expected solution:
```python
def q_sample(x0, t, noise):
    return extract(sqrt_ab, t, x0.shape) * x0 + extract(sqrt_1mab, t, x0.shape) * noise
```

Equivalent acceptable forms:
```python
# Also acceptable — using alpha_bar directly:
ab = alpha_bar[t].reshape(-1, 1, 1, 1)
return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
```

Key checks:
- Both terms present: coefficient × x0 AND coefficient × noise
- Coefficients correctly indexed by `t` (not scalar)
- Sanity-check cell must print `q_sample sanity check passed.`
- Output shape equals x0 shape

| Score | Criteria |
|-------|----------|
| 10 | Correct formula; sanity check passes |
| 7  | Correct formula but coefficients not indexed by t (uses fixed scalar) |
| 5  | One term correct, other missing or wrong coefficient |
| 2  | Attempted — shape correct but wrong formula |
| 0  | `pass` not removed or sanity check crashes |

---

## Part 7 — DDPM: Training Step (10 pts)

**cell-26**

Expected solution:
```python
t     = torch.randint(0, Config.T, (imgs.size(0),), device=device, dtype=torch.long)
noise = torch.randn_like(imgs)
x_t   = q_sample(imgs, t, noise)
pred  = unet(x_t, t, labels)
loss  = F.mse_loss(pred, noise)
opt_ddpm.zero_grad(); loss.backward(); opt_ddpm.step()
```

Key checks:
- `t` randomly sampled per image (not a fixed timestep)
- `q_sample` called with `imgs` (not x_t, not noise)
- MSE is between `pred` and `noise` (predicting noise, not x0)
- Training loss is printed every 10 epochs

| Score | Criteria |
|-------|----------|
| 10 | Correct; loss decreases and is printed |
| 7  | Correct but MSE computed between pred and x0 instead of noise |
| 5  | q_sample called correctly but unet inputs wrong order |
| 3  | t not randomised per batch item |
| 0  | loss remains None or fatal error |

---

## Part 8 — Generated Image Visualisation (10 pts)

**cell-20 (cGAN) + cell-28 (DDPM), 5 pts each**

Expected: Each cell produces a 2-row × 4-column grid of generated images with class names as column titles.

| Score | Criteria |
|-------|----------|
| 5 | Grid correct, images rendered, class titles present |
| 3 | Grid rendered but titles missing or wrong class order |
| 2 | Single row or partial grid |
| 1 | Figure created but all images blank/noise |
| 0 | No output or fatal error |

---

## Analysis Questions (20 pts — 5 pts each)

### Question 1 — Training Stability (5 pts)

Award full credit for answers that include ALL of:
1. cGAN: D and G losses oscillate, may not converge monotonically; DDPM: single MSE loss decreases smoothly
2. cGAN is harder because two competing objectives make it unclear if oscillation = instability or normal adversarial dynamics
3. Valid symptoms: mode collapse (G loss drops to zero, D loss rises; or all generated images look identical), vanishing G gradients (G loss stuck high, D loss ~0.693), training divergence (losses explode)

Partial credit:
- 3 pts: Describes curve shapes correctly but no specific failure mode
- 2 pts: Mentions stability difference without specifics
- 1 pt: Minimal relevant content

### Question 2 — Output Quality (5 pts)

Award full credit for:
1. Specific visual comparison for ≥2 classes (e.g., "DDPM produces smoother textures; GAN images appear more uniform/blurry")
2. One concrete architectural or training change per model (e.g., for GAN: spectral normalisation, larger latent dim, more epochs; for DDPM: larger T, cosine schedule, more model capacity)
3. Justified preference with clinical reasoning (DDPM: better diversity, fewer artifacts; GAN: faster inference at deployment)

### Question 3 — Conditioning Mechanism (5 pts)

Award full credit for:
1. Generator: `c` is embedded into 16-dim vector and concatenated with `z` **before** the first fc layer (early input conditioning)
2. UNetDDPM: `c` is embedded into TIME_DIM (128-dim) vector by `self.class_emb`; injected at **every DBlock** via `self.cemb` — 5 DBlocks (e1, e2, bn, d1, d2)
3. UNetDDPM injection is stronger because it conditions at every resolution and at every layer; generator only conditions once at the input

### Question 4 — Clinical Application (5 pts)

Award full credit for:
1. Either model acceptable with strong justification. DDPM preferred for diversity argument; GAN preferred for speed argument. Must give ≥2 reasons.
2. Valid metrics: FID (Fréchet Inception Distance) — measures distributional similarity between real and generated images using Inception features; or "train on synthetic, test on real" classifier accuracy; or MMD
3. Valid risks: distributional shift (synthetic images don't cover all real-world variation → classifier fails on edge cases); mitigation: use mixed real+synthetic data, evaluate on held-out real test set

---

## Final Grade Summary Template

| Section | Max | Score | Notes |
|---------|-----|-------|-------|
| Data loading | 5 | | |
| Sample visualisation | 5 | | |
| Generator.forward() | 15 | | |
| Discriminator.forward() | 10 | | |
| Training step (D + G) | 15 | | |
| q_sample | 10 | | |
| DDPM training step | 10 | | |
| Image visualisation | 10 | | |
| Q1 Training Stability | 5 | | |
| Q2 Output Quality | 5 | | |
| Q3 Conditioning | 5 | | |
| Q4 Clinical Application | 5 | | |
| **Total** | **100** | | |
