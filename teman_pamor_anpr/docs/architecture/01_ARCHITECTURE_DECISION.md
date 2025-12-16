# 🏗️ Architecture Decision Record (ADR)

**Project:** Teman Pamor ANPR  
**Date:** 2025-01-16  
**Status:** ✅ Approved  
**Decision Makers:** ML Engineering Team

---

## 📋 Context & Problem Statement

### **Project Overview:**

Teman Pamor adalah sistem ANPR (Automatic Number Plate Recognition) untuk Bapenda Jawa Barat dengan 4 model AI esensial:

1. **Plate Detector** (YOLOv11) - Detect & localize license plates
2. **Color Classifier** (MobileNetV2) - Classify plate type
3. **OCR Custom** (CRNN) - Read plate text
4. **Anti-Spoofing** (Binary Classifier) - Detect photo vs screen

### **Key Requirements:**

- ✅ **Mobile-first**: Deploy on Android/iOS via Flutter
- ✅ **Multi-model**: 4 independent AI models
- ✅ **On-device**: TFLite inference, no cloud dependency
- ✅ **Small team**: 1-2 developers
- ✅ **Production-ready**: Not a prototype
- ✅ **Portfolio project**: Impress recruiters

### **Problem:**

Existing structure (`plate-recognition/`) is **flat** and **unorganized**:

- ❌ No clear separation between data, training, deployment
- ❌ Hard to manage 4 different models
- ❌ Not scalable for adding new models
- ❌ No production practices (versioning, validation, monitoring)
- ❌ Not impressive for job applications

---

## 🔍 Evaluation of Alternatives

We evaluated **5 industry-standard architectures**:

### **1. Cookiecutter Data Science**

**Score:** 26/50 ❌

**Pros:**

- ✅ Simple & widely known
- ✅ Good for research projects
- ✅ Easy to learn

**Cons:**

- ❌ **Not mobile-first** - No consideration for TFLite deployment
- ❌ **Single-model focused** - Hard to manage 4 models
- ❌ **Research-oriented** - No production practices
- ❌ **No deployment layer** - Just notebooks & training

**Verdict:** ❌ **Rejected** - Too academic, not production-focused

---

### **2. Google MLOps (Full Stack)**

**Score:** 29/50 ❌

**Pros:**

- ✅ Industry gold standard
- ✅ Production-grade
- ✅ Excellent for multi-model
- ✅ CI/CD ready

**Cons:**

- ❌ **Over-engineered** - Needs K8s, Airflow, MLflow, Kubeflow
- ❌ **Not mobile-first** - Designed for cloud serving
- ❌ **Team size mismatch** - Requires 5+ person team
- ❌ **Infrastructure heavy** - Need DevOps expertise

**Verdict:** ❌ **Rejected** - Overkill for 1-2 developer team

---

### **3. Kedro Framework**

**Score:** 28/50 ❌

**Pros:**

- ✅ Modular pipelines
- ✅ Reproducibility focus
- ✅ Good for teams

**Cons:**

- ❌ **Framework lock-in** - Must use Kedro
- ❌ **Not mobile-optimized** - No TFLite considerations
- ❌ **Learning curve** - Need to learn Kedro paradigm
- ❌ **Deployment unclear** - Focus on training, not serving

**Verdict:** ❌ **Rejected** - Good framework, but not right fit

---

### **4. Mobile ML (TFLite Pattern)**

**Score:** 37/50 ✅

**Pros:**

- ✅ **Mobile-first design** - Everything for on-device
- ✅ **Lightweight** - No heavy infrastructure
- ✅ **TFLite focused** - Conversion, quantization, validation
- ✅ **Small team friendly** - Simple & clear

**Cons:**

- ❌ **Multi-model support weak** - Not designed for 4+ models
- ❌ **No data platform** - Weak labeling/data management
- ❌ **No monitoring** - Production observability missing
- ❌ **Not production-grade** - Missing versioning, CI/CD

**Verdict:** ⚠️ **Close, but incomplete** - Good foundation, needs enhancement

---

### **5. Modular Monolith (Uber/Airbnb)**

**Score:** 31/50 ❌

**Pros:**

- ✅ **Excellent multi-model** - Platform thinking
- ✅ **Production-grade** - All best practices
- ✅ **Scalable** - 1 to 100+ models
- ✅ **Enterprise-ready** - Governance, monitoring

**Cons:**

- ❌ **Extreme complexity** - Need dedicated ML platform team
- ❌ **Not mobile-optimized** - Cloud-first, not edge-first
- ❌ **Overkill** - Designed for unicorn scale
- ❌ **Infrastructure cost** - High maintenance

**Verdict:** ❌ **Rejected** - Amazing, but way too complex

---

## ✅ Decision: Hybrid "Mobile-First MLOps"

**Score:** 43/50 ⭐ **WINNER**

### **Why Hybrid?**

We created a **custom hybrid** combining:

- **Mobile ML Pattern** (foundation) → Mobile-first, TFLite focus
- **Google MLOps** (selected practices) → Versioning, validation, monitoring
- **Modular thinking** (platform layer) → Multi-model support, shared components

### **Key Design Decisions:**

#### **1. Platform Layer (`00_platform/`)** 🛠️

**Decision:** Add shared platform layer for common functionality

**Rationale:**

- ✅ **DRY principle** - Don't repeat data validation, preprocessing across 4 models
- ✅ **Consistency** - All models use same quality standards
- ✅ **Maintainability** - Fix bug once, benefits all models
- ✅ **Portfolio impact** - Shows system thinking, not just "training scripts"

**Implementation:**

```python
# Example: Shared preprocessing
from platform.preprocessing import normalize_image, augment_batch

# Used by all 4 models
image = normalize_image(raw_image)
batch = augment_batch(images, augmentation_config)
```

---

#### **2. Data Platform (`01_data_platform/`)** 📊

**Decision:** Separate data concerns from training

**Rationale:**

- ✅ **Data quality gates** - Validate before training
- ✅ **Organized by stage** - raw → validated → augmented
- ✅ **Labeling tools** - Active development area
- ✅ **Reproducibility** - Track data lineage

**Structure:**

```
01_data_platform/
├── labeling_tools/     # Interactive GUI
├── datasets/
│   ├── 00_raw/         # Immutable source
│   ├── 01_validated/   # Quality checked
│   └── 02_augmented/   # Training ready
└── notebooks/          # EDA
```

---

#### **3. Training Platform (`02_training_platform/`)** 🤖

**Decision:** One folder per model, identical structure

**Rationale:**

- ✅ **Multi-model native** - 4 models, scalable to 10+
- ✅ **Independence** - Each model evolves separately
- ✅ **Consistency** - Same folder structure, easy navigation
- ✅ **Experiment tracking** - MLflow-style experiment organization

**Per-Model Structure:**

```
01_plate_detector_yolo/
├── experiments/        # Versioned experiments
│   ├── exp_001_baseline/
│   └── exp_002_improved/
├── configs/           # Hyperparameters
├── src/              # Training code
├── notebooks/        # Analysis
└── README.md         # Model docs
```

**Benefits:**

- New engineer? Look at one model, understand all models
- Add model #5? Copy structure, fill in code
- Compare experiments? Same folder layout

---

#### **4. Deployment Platform (`03_deployment_platform/`)** 📱

**Decision:** Mobile-first with strict validation pipeline

**Rationale:**

- ✅ **Mobile constraints** - Size (<10MB), latency (<200ms), battery
- ✅ **Quality gates** - Can't deploy without validation
- ✅ **Versioning** - Track model versions in production
- ✅ **Flutter ready** - Direct integration code

**Deployment Flow:**

```
Training → Conversion → Validation → Staging → Production
   (02)       (03)         (03)        (03)       (03)
```

**Validation Checks:**

1. ✅ Accuracy test (accuracy drop < 2%)
2. ✅ Latency benchmark (< 200ms)
3. ✅ Size check (< 10MB per model)
4. ✅ Compatibility test (TFLite ops supported)

**Only after ALL pass → Deploy to `mobile_models/v1.x.x/`**

---

#### **5. CI/CD (`04_ci_cd/`)** 🔄

**Decision:** Automate everything, but start simple

**Rationale:**

- ✅ **Reproducibility** - Same environment every time
- ✅ **Quality gates** - Automated testing
- ✅ **Portfolio value** - Shows production mindset
- ✅ **Team efficiency** - Less manual work

**GitHub Actions Workflows:**

```yaml
01_data_validation.yml    # Trigger: New data uploaded
02_model_training.yml     # Trigger: Manual or schedule
03_model_validation.yml   # Trigger: Training complete
04_model_deployment.yml   # Trigger: Validation passed
```

**Start Simple:**

- Phase 1: Manual scripts (week 1)
- Phase 2: GitHub Actions (week 2-3)
- Phase 3: Full automation (week 4+)

---

## 📊 Comparison: Before vs After

### **Before (Current Structure):**

```
plate-recognition/
├── label_tool.py
├── train_color_classifier.py
├── train_native_rotation.py
├── dataset/
├── runs/
└── models/
```

**Problems:**

- ❌ Flat structure - everything mixed
- ❌ No clear boundaries
- ❌ Hard to add new model
- ❌ No production practices

### **After (Hybrid Architecture):**

```
teman_pamor_anpr/
├── 00_platform/           # Shared layer
├── 01_data_platform/      # Data concerns
├── 02_training_platform/  # Training (4 models)
├── 03_deployment_platform/# Mobile deployment
└── 04_ci_cd/             # Automation
```

**Benefits:**

- ✅ Clear separation of concerns
- ✅ Multi-model ready (4 now, 10+ future)
- ✅ Production practices built-in
- ✅ Impressive for portfolio

---

## 🎯 Success Metrics

We'll measure success by:

### **1. Developer Experience:**

- ✅ New model can be added in < 1 day
- ✅ New team member onboards in < 1 week
- ✅ Common tasks have `make` commands

### **2. Production Readiness:**

- ✅ All models versioned (`v1.0.0`, `v1.1.0`)
- ✅ 100% models pass validation before deployment
- ✅ <200ms end-to-end latency on mobile

### **3. Portfolio Impact:**

- ✅ Recruiters understand structure in <5 minutes
- ✅ README clearly shows production thinking
- ✅ Architecture diagrams in docs/

---

## 🚀 Implementation Plan

### **Phase 1: Foundation (Week 1)**

- ✅ Create directory structure
- ✅ Migrate existing files
- ✅ Write documentation

### **Phase 2: Platform Layer (Week 2)**

- ✅ Implement shared data validation
- ✅ Implement shared preprocessing
- ✅ Setup model registry

### **Phase 3: Training Platform (Week 3)**

- ✅ Refactor 4 models to new structure
- ✅ Add experiment tracking
- ✅ Standardize configs

### **Phase 4: Deployment Pipeline (Week 4)**

- ✅ Build conversion scripts
- ✅ Build validation pipeline
- ✅ Setup model versioning

### **Phase 5: CI/CD (Week 5+)**

- ✅ GitHub Actions workflows
- ✅ Automated testing
- ✅ Deployment automation

---

## 📚 References

### **Industry Examples:**

1. **Uber Michelangelo** - ML Platform architecture
2. **Google TFX** - TensorFlow Extended best practices
3. **Airbnb Bighead** - ML infrastructure
4. **MLOps Community** - Best practices

### **Academic:**

1. "Hidden Technical Debt in Machine Learning Systems" (Google, 2015)
2. "Machine Learning: The High-Interest Credit Card of Technical Debt" (Sculley et al.)
3. "MLOps: Continuous delivery and automation pipelines in machine learning" (Google Cloud)

### **Frameworks Evaluated:**

1. Cookiecutter Data Science - drivendata.github.io/cookiecutter-data-science
2. Kedro - kedro.org
3. TensorFlow Extended (TFX) - tensorflow.org/tfx
4. MLflow - mlflow.org

---

## 🤔 Trade-offs & Risks

### **Complexity vs Simplicity:**

- **Trade-off:** More folders than flat structure
- **Mitigation:** Clear documentation, consistent patterns
- **Risk:** Low - Structure is intuitive

### **Initial Setup Time:**

- **Trade-off:** 1 week to setup vs immediate coding
- **Mitigation:** Automated migration script
- **Risk:** Medium - Worth investment for long-term

### **Learning Curve:**

- **Trade-off:** Need to understand architecture
- **Mitigation:** Excellent documentation, onboarding guide
- **Risk:** Low - Architecture is self-explanatory

---

## ✅ Conclusion

**Decision:** ✅ **APPROVED**

We will implement **Hybrid "Mobile-First MLOps"** architecture because:

1. ✅ **Best fit for requirements** - Mobile-first, multi-model, small team
2. ✅ **Production-ready** - Versioning, validation, monitoring
3. ✅ **Scalable** - Easy to add models 5, 6, 7...
4. ✅ **Portfolio impressive** - Shows engineering excellence
5. ✅ **Right complexity** - Not over/under-engineered

**Next Step:** Execute migration with `migrate_project_structure.py`

---

**Approved by:** ML Engineering Team  
**Date:** 2025-01-16  
**Version:** 1.0
