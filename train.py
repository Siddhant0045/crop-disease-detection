import os
import json
import re
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# ✅ CONFIG (UPDATE THESE)
# =========================
DATA_DIR = "dataset/PlantVillage"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 123

# ✅ Training plan
EPOCHS_FROZEN = 10      # head training
EPOCHS_FINETUNE = 6     # finetune 

RESUME_FROM_LAST = True  # resume from latest model_XX.keras

# MobileNetV2 preprocessing
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input


# =========================
# 1) LOAD DATA
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_ds.class_names
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

num_classes = len(class_names)

# Performance pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


# =========================
# 2) Version helpers
# =========================
def get_last_saved_model(model_dir: str):
    files = [f for f in os.listdir(model_dir) if re.match(r"model_\d{2}\.keras$", f)]
    if not files:
        return None, 0
    files.sort()
    latest = files[-1]
    latest_epoch = int(latest.replace("model_", "").replace(".keras", ""))
    return os.path.join(model_dir, latest), latest_epoch


# =========================
# 3) Build / Load Model
# =========================
start_epoch = 0
model = None

if RESUME_FROM_LAST:
    last_path, last_epoch = get_last_saved_model(MODEL_DIR)
    if last_path:
        print(f"✅ Resuming from {last_path} (trained till epoch {last_epoch})")
        model = tf.keras.models.load_model(last_path)
        start_epoch = last_epoch
    else:
        print("ℹ️ No saved model found. Starting fresh...")

if model is None:
    # Augmentation
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augmentation")

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    # Full model with correct preprocessing
    inputs = layers.Input(shape=(224, 224, 3))
    x = aug(inputs)
    x = preprocess(x)  
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

# Compile (safe even for loaded model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# 4) Phase 1: Train Head (Frozen Base)
# =========================
target_epoch_phase1 = max(start_epoch, 0) + EPOCHS_FROZEN
print(f"\n🚀 Phase 1 (Frozen base): train until epoch {target_epoch_phase1}")

for epoch in range(start_epoch + 1, target_epoch_phase1 + 1):
    print(f"\n🧠 Epoch {epoch} (Frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

    save_path = os.path.join(MODEL_DIR, f"model_{epoch:02d}.keras")
    model.save(save_path)
    print(f"💾 Saved: {save_path}")


# =========================
# 5) Phase 2: Fine-tune (Unfreeze last layers)
# =========================
# Find base model inside loaded model (works for both fresh & resumed)
base = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2"):
        base = layer
        break

if base is None:
    # in case it got wrapped differently
    base = model.get_layer(index=3) if len(model.layers) > 3 else None

if base is not None:
    base.trainable = True

    # Unfreeze only last N layers (avoid overfitting)
    UNFREEZE_LAST = 40
    for layer in base.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start_epoch_ft = target_epoch_phase1
    target_epoch_phase2 = start_epoch_ft + EPOCHS_FINETUNE

    print(f"\n🔥 Phase 2 (Fine-tune): train until epoch {target_epoch_phase2}")

    for epoch in range(start_epoch_ft + 1, target_epoch_phase2 + 1):
        print(f"\n🧠 Epoch {epoch} (Fine-tune)")
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

        save_path = os.path.join(MODEL_DIR, f"model_{epoch:02d}.keras")
        model.save(save_path)
        print(f"💾 Saved: {save_path}")
else:
    print("⚠️ Could not locate base MobileNetV2 layer for fine-tuning. Skipping Phase 2.")
