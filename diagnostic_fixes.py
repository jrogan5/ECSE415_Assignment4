# ================================
# DIAGNOSTIC CELL - Add this to your notebook NOW
# Run this to understand what's wrong
# ================================

print("=" * 70)
print("TRAINING DIAGNOSTIC REPORT")
print("=" * 70)

# 1. Check current training status
print(f"\n1. TRAINING STATUS:")
print(f"   Current epoch: {epoch+1}/{num_epochs_v2}")
print(f"   Current LR: {optimizer_v2.param_groups[0]['lr']:.2e}")

# 2. Check recent loss history
print(f"\n2. LOSS HISTORY (last 10 epochs):")
if len(val_losses_v2) >= 10:
    recent_losses = val_losses_v2[-10:]
    for i, loss in enumerate(recent_losses):
        epoch_num = len(val_losses_v2) - 10 + i + 1
        print(f"   Epoch {epoch_num}: {loss:.2f}")
else:
    for i, loss in enumerate(val_losses_v2):
        print(f"   Epoch {i+1}: {loss:.2f}")

# 3. Check if loss is improving
print(f"\n3. LOSS TREND:")
if len(val_losses_v2) >= 10:
    first_10_avg = np.mean(val_losses_v2[:10])
    last_10_avg = np.mean(val_losses_v2[-10:])
    improvement = ((first_10_avg - last_10_avg) / first_10_avg) * 100
    print(f"   First 10 epochs avg: {first_10_avg:.2f}")
    print(f"   Last 10 epochs avg: {last_10_avg:.2f}")
    print(f"   Improvement: {improvement:.1f}%")

    if improvement < 10:
        print("   ⚠️  WARNING: Less than 10% improvement - MODEL IS STUCK!")
    elif improvement < 30:
        print("   ⚠️  CAUTION: Slow learning - consider restarting")
    else:
        print("   ✓ Model is learning (but maybe slowly)")

# 4. Check for common issues
print(f"\n4. POTENTIAL ISSUES:")

# Check if LR is too low
current_lr = optimizer_v2.param_groups[0]['lr']
if current_lr < 1e-5:
    print("   ⚠️  Learning rate is VERY LOW (< 1e-5) - model may be stuck")

# Check if loss is abnormally high
min_loss = min(val_losses_v2)
current_loss = val_losses_v2[-1]
print(f"   Min loss so far: {min_loss:.2f}")
print(f"   Current loss: {current_loss:.2f}")

if current_loss > 1000:
    print("   ⚠️  CRITICAL: Loss > 1000 is extremely high for RetinaNet")
    print("   This suggests:")
    print("      - Model is not learning the task properly")
    print("      - Learning rate might have been too high initially OR too low now")
    print("      - Data preprocessing might have issues")

# 5. Check data sample
print(f"\n5. CHECKING DATA SANITY:")
for images, targets in val_loader_v2:
    print(f"   Batch size: {len(images)}")
    print(f"   Image tensor shapes: {[img.shape for img in images[:2]]}")
    print(f"   Number of boxes in first sample: {len(targets[0]['boxes'])}")
    print(f"   Labels in first sample: {targets[0]['labels'].tolist()}")
    print(f"   Box coordinates (first sample):")
    if len(targets[0]['boxes']) > 0:
        box = targets[0]['boxes'][0]
        print(f"      First box (xyxy): [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        width = box[2] - box[0]
        height = box[3] - box[1]
        print(f"      Box dimensions: {width:.1f} x {height:.1f}")
        if width <= 0 or height <= 0:
            print("      ⚠️  CRITICAL: Invalid box detected!")
    break

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if current_loss > 1000 and (epoch+1) > 30:
    print("❌ STOP TRAINING - Model is not converging")
    print("\nYou should:")
    print("1. RESTART with aggressive learning rate schedule")
    print("2. OR try a different model architecture")
    print("3. Check data preprocessing carefully")
else:
    print("⚠️  Training might still work but needs optimization")
    print("Consider the alternative configs below")

# ================================
# ALTERNATIVE 1: Aggressive Restart with Higher LR
# ================================

print("\n" + "=" * 70)
print("ALTERNATIVE 1: AGGRESSIVE LR RESTART")
print("Use this if you want to restart training with more aggressive learning")
print("=" * 70)

alternative_1_code = '''
# ================================
# Alternative 1: Aggressive Learning Rate Restart
# ================================

# Re-initialize model
model_retina_aggressive = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)

cls_head = model_retina_aggressive.head.classification_head
in_channels = cls_head.cls_logits.in_channels
num_anchors = cls_head.num_anchors
new_num_classes = num_classes + 1

cls_head.cls_logits = nn.Conv2d(
    in_channels,
    num_anchors * new_num_classes,
    kernel_size=3,
    stride=1,
    padding=1
)
cls_head.num_classes = new_num_classes
model_retina_aggressive.to(device)

# Aggressive optimizer with HIGHER initial LR
optimizer_aggressive = torch.optim.AdamW(
    model_retina_aggressive.parameters(),
    lr=5e-4,  # 2.5x higher than before
    weight_decay=1e-4
)

num_epochs_aggressive = 50

# Cosine Annealing for smooth decay
lr_scheduler_aggressive = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_aggressive,
    T_max=num_epochs_aggressive,
    eta_min=1e-6
)

# Warmup for first 5 epochs
warmup_epochs = 5
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer_aggressive,
    start_factor=0.1,
    total_iters=warmup_epochs
)

# Combine warmup + cosine
lr_scheduler_aggressive = torch.optim.lr_scheduler.SequentialLR(
    optimizer_aggressive,
    schedulers=[warmup_scheduler, lr_scheduler_aggressive],
    milestones=[warmup_epochs]
)

train_losses_aggressive = []
val_losses_aggressive = []

print(f"Starting AGGRESSIVE training with LR={optimizer_aggressive.param_groups[0]['lr']:.2e}")
print("This uses Cosine Annealing with warmup for stable, aggressive learning")

for epoch in range(num_epochs_aggressive):
    # Train
    model_retina_aggressive.train()
    running_train_loss = 0.0

    for images, targets in train_loader_v2:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model_retina_aggressive(images, targets)
        losses = sum(loss_dict.values())

        optimizer_aggressive.zero_grad()
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_retina_aggressive.parameters(), max_norm=1.0)

        optimizer_aggressive.step()
        running_train_loss += losses.item()

    avg_train_loss = running_train_loss / max(1, len(train_loader_v2))
    train_losses_aggressive.append(avg_train_loss)

    # Validation
    running_val_loss = 0.0
    with torch.no_grad():
        model_retina_aggressive.train()
        for images, targets in val_loader_v2:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model_retina_aggressive(images, targets)
            losses = sum(loss_dict.values())
            running_val_loss += losses.item()

    avg_val_loss = running_val_loss / max(1, len(val_loader_v2))
    val_losses_aggressive.append(avg_val_loss)

    lr_scheduler_aggressive.step()
    current_lr = optimizer_aggressive.param_groups[0]['lr']

    if (epoch + 1) % 5 == 0 or epoch < 5:
        print(f"Epoch [{epoch+1}/{num_epochs_aggressive}]  "
              f"Train: {avg_train_loss:.4f}  "
              f"Val: {avg_val_loss:.4f}  "
              f"LR: {current_lr:.6f}")

        # Early warning system
        if epoch >= 10 and avg_val_loss > 1000:
            print(f"   ⚠️  WARNING: Loss still high at epoch {epoch+1}")

print("\\nAggressive training complete!")
print(f"Final val loss: {val_losses_aggressive[-1]:.4f}")
print(f"Best val loss: {min(val_losses_aggressive):.4f} at epoch {np.argmin(val_losses_aggressive) + 1}")
'''

print("Copy and run this code in a new cell:")
print(alternative_1_code)

# ================================
# ALTERNATIVE 2: Faster RCNN
# ================================

print("\n" + "=" * 70)
print("ALTERNATIVE 2: SWITCH TO FASTER RCNN")
print("Use this if RetinaNet fundamentally doesn't work for your data")
print("=" * 70)

alternative_2_code = '''
# ================================
# Alternative 2: Faster RCNN (Two-Stage Detector)
# ================================

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Initialize Faster RCNN with COCO pretrained weights
weights_frcnn = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model_frcnn = fasterrcnn_resnet50_fpn(weights=weights_frcnn)

# Replace the box predictor head
in_features = model_frcnn.roi_heads.box_predictor.cls_score.in_features
model_frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)

model_frcnn.to(device)

# Optimizer with moderate learning rate
optimizer_frcnn = torch.optim.SGD(
    model_frcnn.parameters(),
    lr=5e-3,  # Faster RCNN often works better with SGD + higher LR
    momentum=0.9,
    weight_decay=5e-4
)

num_epochs_frcnn = 40

# Step LR with earlier decay
lr_scheduler_frcnn = torch.optim.lr_scheduler.StepLR(
    optimizer_frcnn,
    step_size=15,  # Decay every 15 epochs
    gamma=0.1
)

train_losses_frcnn = []
val_losses_frcnn = []

print(f"Training Faster RCNN with LR={optimizer_frcnn.param_groups[0]['lr']:.2e}")
print("Faster RCNN is a two-stage detector - might work better for small signs")

for epoch in range(num_epochs_frcnn):
    # Train
    model_frcnn.train()
    running_train_loss = 0.0

    for images, targets in train_loader_v2:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model_frcnn(images, targets)
        losses = sum(loss_dict.values())

        optimizer_frcnn.zero_grad()
        losses.backward()
        optimizer_frcnn.step()

        running_train_loss += losses.item()

    avg_train_loss = running_train_loss / max(1, len(train_loader_v2))
    train_losses_frcnn.append(avg_train_loss)

    # Validation
    running_val_loss = 0.0
    with torch.no_grad():
        model_frcnn.train()
        for images, targets in val_loader_v2:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model_frcnn(images, targets)
            losses = sum(loss_dict.values())
            running_val_loss += losses.item()

    avg_val_loss = running_val_loss / max(1, len(val_loader_v2))
    val_losses_frcnn.append(avg_val_loss)

    lr_scheduler_frcnn.step()
    current_lr = optimizer_frcnn.param_groups[0]['lr']

    if (epoch + 1) % 5 == 0 or epoch < 5:
        print(f"Epoch [{epoch+1}/{num_epochs_frcnn}]  "
              f"Train: {avg_train_loss:.4f}  "
              f"Val: {avg_val_loss:.4f}  "
              f"LR: {current_lr:.6f}")

print("\\nFaster RCNN training complete!")
print(f"Final val loss: {val_losses_frcnn[-1]:.4f}")
print(f"Best val loss: {min(val_losses_frcnn):.4f} at epoch {np.argmin(val_losses_frcnn) + 1}")

# Save model
torch.save(model_frcnn.state_dict(), "model_faster_rcnn.pth")
'''

print("Copy and run this code in a new cell:")
print(alternative_2_code)

print("\n" + "=" * 70)
print("FINAL RECOMMENDATION FOR YOUR SITUATION")
print("=" * 70)
print("""
Given you're at epoch 41/60 with loss ~1700:

1. STOP your current training immediately (it's wasting time)

2. Try Alternative 1 FIRST (Aggressive LR Restart):
   - Uses Cosine Annealing (smoother than MultiStepLR)
   - Higher initial LR (5e-4 vs 2e-4)
   - Warmup prevents instability
   - Should see loss < 500 by epoch 20

3. If Alternative 1 fails (loss > 800 at epoch 20):
   - Try Alternative 2 (Faster RCNN)
   - Uses SGD optimizer (sometimes better for detection)
   - Different architecture might work better

4. Expected good training behavior:
   - Epoch 1-5: Loss drops from ~2000 to ~800
   - Epoch 5-15: Loss drops to ~300-500
   - Epoch 15-30: Loss drops to ~100-200
   - Epoch 30+: Loss stabilizes around 50-150

If you don't see this pattern by epoch 15, something is wrong!
""")
