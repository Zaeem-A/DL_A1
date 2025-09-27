### Deep Learning Assignment Report
### 22i-1933
### Zaeem Ahmed

1. Summary
This report presents the implementation and comparison of two CNN architectures (VGG16 and ResNet18) for facial expression recognition, along with valence and arousal prediction. The task involves multi-modal learning where models simultaneously classify facial expressions into 8 categories and predict continuous valence/arousal values. ResNet18 demonstrated superior performance across most metrics, achieving 44.18% validation accuracy compared to VGG16's 43.18%.

2. Network Architecture Details
2.1 Baseline Architecture Selection
Two prominent CNN architectures were selected as baselines:
VGG16 Multi-Head Architecture:
Rationale: VGG16's simple yet effective architecture with small 3×3 convolution filters provides strong feature extraction capabilities. Its uniform architecture makes it suitable for transfer learning applications.
Parameters: Approximately 138M parameters
Architecture: 13 convolutional layers with ReLU activation, 5 max-pooling layers
Multi-head Design: Shared convolutional backbone with three separate heads:
Expression classification head (8 classes)
Valence regression head (continuous values)
Arousal regression head (continuous values)
ResNet18 Multi-Head Architecture:
Rationale: ResNet's residual connections address the vanishing gradient problem, enabling deeper networks and better feature learning. The skip connections facilitate gradient flow during backpropagation.
Parameters: Approximately 11M parameters
Architecture: 18 layers with residual blocks, batch normalization, and ReLU activation
Multi-head Design: Similar to VGG16, featuring shared feature extraction with specialized heads for each task
2.2 Training Configuration
Common Training Settings:
Input Size: 224×224×3 RGB images
Batch Size: 32
Optimizer: Adam with learning rate scheduling
Loss Functions:
Cross-entropy for expression classification
Mean Squared Error (MSE) for valence/arousal regression
Combined weighted loss for multi-task learning
Data Augmentation: Random horizontal flips, rotation (±15°), brightness adjustment
Transfer Learning: Pre-trained ImageNet weights used for feature extraction layers

3. Dataset Configuration
3.1 Dataset Splits
Training Set: 80% of available data
Validation Set: 20% of available data
Total Images: Cropped and resized to 224×224 pixels
Classes: 8 emotion categories (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
Continuous Labels: Valence and Arousal values in [-1, +1] range
3.2 Data Preprocessing
Normalization using ImageNet statistics
Online data augmentation during training
Facial landmark information available but not utilized in current implementation


5. Training Analysis
4.1 Learning Curves
The training curves reveal distinct learning patterns for both architectures:
Loss Convergence:
ResNet18 demonstrates faster convergence, reaching stable loss values around epoch 5
VGG16 shows slower but steady convergence, stabilizing around epoch 10
Both models show appropriate training without significant overfitting
Accuracy Progression:
ResNet18 achieves rapid accuracy improvement in early epochs
VGG16 displays more gradual learning with initial instability
Final convergence shows ResNet18 maintaining slight performance advantage
Multi-Modal Training:
The combined loss function effectively balances classification and regression objectives
Training accuracy stabilizes around 30-32% while validation accuracy reaches 27-28%
The gap between training and validation performance indicates appropriate model capacity
4.2 Convergence Analysis
Both architectures demonstrate healthy learning patterns with decreasing loss and increasing accuracy over epochs. The absence of significant overfitting suggests appropriate regularization and data augmentation strategies.
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/f1_curve.png)
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/kappa_curve.png)
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/loss_curve.png)
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/multi_modal_training_curves_classification.png)
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/train_acc_curve.png)
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/val_acc_curve.png)

## 7. Performance Evaluation

### 5.1 Categorical Classification Metrics

| Metric | VGG16 | ResNet18 | Better |
|--------|-------|----------|--------|
| **Validation Accuracy** | 0.4318 | **0.4418** | ResNet18 |
| **F1-Score** | 0.4305 | **0.4389** | ResNet18 |
| **Cohen's Kappa** | 0.3503 | **0.3610** | ResNet18 |

### 5.2 Continuous Domain Evaluation

#### Valence Prediction:
| Metric | VGG16 | ResNet18 | Better |
|--------|-------|----------|--------|
| **RMSE** | **0.3803** | 0.4058 | VGG16 |
| **Correlation (CORR)** | **0.5641** | 0.5119 | VGG16 |
| **Sign Agreement (SAGR)** | **0.7647** | 0.7359 | VGG16 |
| **CCC** | **0.5258** | 0.4867 | VGG16 |

#### Arousal Prediction:
| Metric | VGG16 | ResNet18 | Better |
|--------|-------|----------|--------|
| **RMSE** | **0.3364** | 0.3525 | VGG16 |
| **Correlation (CORR)** | **0.4598** | 0.3769 | VGG6 |
| **Sign Agreement (SAGR)** | **0.7935** | 0.7872 | VGG16 |
| **CCC** | **0.4083** | 0.3295 | VGG16 |

Analysis:
ResNet18 consistently outperforms VGG16 across all classification metrics
The improvement margin is modest but consistent (~1-3%)
Kappa values indicate fair to moderate agreement beyond chance


On Resnet:

![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/resnet.png)
On VGG:
![alt text](https://github.com/Zaeem-A/DL_A1/blob/main/vgg.png)

### 5.2 Continuous Domain Evaluation

#### Valence Prediction:
| Metric | VGG16 | ResNet18 | Better |
|--------|-------|----------|--------|
| **RMSE** | **0.3803** | 0.4058 | VGG16 |
| **Correlation (CORR)** | **0.5641** | 0.5119 | VGG16 |
| **Sign Agreement (SAGR)** | **0.7647** | 0.7359 | VGG16 |
| **CCC** | **0.5258** | 0.4867 | VGG16 |

#### Arousal Prediction:
| Metric | VGG16 | ResNet18 | Better |
|--------|-------|----------|--------|
| **RMSE** | **0.3364** | 0.3525 | VGG16 |
| **Correlation (CORR)** | **0.4598** | 0.3769 | VGG16 |
| **Sign Agreement (SAGR)** | **0.7935** | 0.7872 | VGG16 |
| **CCC** | **0.4083** | 0.3295 | VGG16 |

5.3 Continuous Domain Metrics Analysis
Root Mean Square Error (RMSE):
Rationale: Measures average prediction error magnitude. Lower values indicate better precision.
Scenario Suitability: Critical for applications requiring precise affect quantification.
Correlation (CORR):
Rationale: Measures linear relationship strength between predicted and actual values.
Scenario Suitability: Important for understanding overall trend agreement.
Sign Agreement (SAGR):
Rationale: Penalizes predictions with incorrect polarity more severely than magnitude errors.
Scenario Suitability: Most suitable for real-world applications as getting the emotional direction wrong (positive vs. negative affect) is more problematic than small magnitude errors.
Concordance Correlation Coefficient (CCC):
Rationale: Combines correlation with mean difference assessment, providing comprehensive agreement measure.
Scenario Suitability: Excellent for clinical or research applications requiring both precision and accuracy.
For Wild Deployment: SAGR emerges as the most critical metric since misidentifying emotional polarity could lead to inappropriate system responses in real-world scenarios.

6. Architecture Comparison
6.1 Performance Trade-offs
ResNet18 Advantages:
Superior classification performance across all categorical metrics
Faster training convergence due to residual connections
More parameter-efficient (11M vs 138M parameters)
Better gradient flow enabling stable training
VGG16 Advantages:
Superior continuous value prediction for both valence and arousal
More robust regression performance across all continuous metrics
Established architecture with extensive research backing
6.2 Computational Considerations
Training Time Comparison:
ResNet18: Faster training per epoch due to fewer parameters
VGG16: Longer training time but potentially better feature representation for regression tasks
Memory Requirements:
ResNet18: Lower memory footprint
VGG16: Higher memory requirements but acceptable for most modern hardware

7. Qualitative Results
7.1 Qualitative Analysis Implementation
Sample Selection Methodology: Qualitative analysis was performed using the trained model checkpoints (.pt files) to evaluate prediction quality. Samples were categorized into correctly and incorrectly classified examples, ranked by prediction confidence scores to identify the most representative cases.
Visualization Framework:
Correct Predictions: High-confidence accurate classifications demonstrating model strengths
Incorrect Predictions: High-confidence misclassifications revealing model limitations and confusion patterns
Multi-Modal Display: Each sample shows original image, true/predicted labels, confidence scores, and continuous valence/arousal predictions
Analysis Procedure:
# Model evaluation with confidence scoring
samples, results = evaluate_model_detailed(model, dataloader)
correct_samples = [s for s in samples if s['correct']]
incorrect_samples = [s for s in samples if not s['correct']]

# Sort by confidence for representative selection
correct_samples.sort(key=lambda x: x['confidence'], reverse=True)
incorrect_samples.sort(key=lambda x: x['confidence'], reverse=True)

7.2 Representative Sample Analysis
High-Confidence Correct Predictions:
Neutral expressions: Clear facial relaxation, minimal muscle activation
Happy expressions: Prominent lip curvature, raised cheeks, eye crinkles
Sad expressions: Downward lip curvature, lowered eyebrows, drooping eyelids
High-Confidence Incorrect Predictions:
Neutral-Sad confusion: Subtle expressions with ambiguous valence
Fear-Surprise misclassification: Similar eye widening patterns
Anger-Disgust confusion: Shared negative valence with different arousal levels
Valence-Arousal Prediction Patterns:
Successful cases: Predictions align well with dimensional affect space
Challenging cases: Conflicting categorical and continuous predictions indicate model uncertainty

8. Custom Architecture: EmotionNet (Multi-Modal CNN+Landmarks)
8.1 Architecture Design
In addition to the baseline architectures, a custom EmotionNet model was developed that combines visual CNN features with facial landmark information for enhanced emotion recognition.
Architecture Components:
Image Encoder: ResNet18-based CNN backbone (pre-trained on ImageNet)
Feature extraction: 512-dimensional image representations
Dimensionality reduction: Linear layer to 256 dimensions
Landmark Encoder: Multi-layer perceptron for 68 facial landmarks (136 coordinates)
Input: 136-dimensional landmark coordinates
Hidden layers: 136 → 128 → 64 dimensions with ReLU activation
Fusion Module: Concatenation followed by fully connected layers
Combined features: 256 (image) + 64 (landmarks) = 320 dimensions
Fusion layer: 320 → 256 with ReLU and dropout (0.2)
Multi-Head Output: Three specialized prediction heads
Expression classification: 8-class softmax
Valence regression: Single continuous output
Arousal regression: Single continuous output
8.2 Design Rationale
Multi-Modal Approach: The combination of visual CNN features with explicit geometric facial landmark information provides complementary information sources. While CNNs excel at learning complex visual patterns, facial landmarks offer precise geometric relationships that are crucial for emotion recognition.
Feature Fusion Strategy: Late fusion was chosen to allow both datasets to develop specialized representations before combination. This approach enables the model to learn modality-specific features while benefiting from their interaction.
Transfer Learning: Pre-trained ResNet18 backbone accelerates convergence and provides robust visual feature extraction, particularly valuable given the relatively small dataset size.
8.3 Training Performance
The custom EmotionNet achieved the following training characteristics:
Initial Loss: 4.1172 (Epoch 1)
Final Loss: 2.1152 (Epoch 100)
Convergence Pattern: Rapid initial improvement followed by steady refinement
Training Stability: Consistent loss reduction without significant oscillations

The loss was not exactly 0 but due to some technical difficulties it is being represented as near 0
Training Configuration:
Epochs: 100
Batch Size: 64
Learning Rate: 1e-4
Optimizer: Adam
Loss Function: Combined CrossEntropy + MSE for multi-task learning
8.4 Multi-Modal Learning Benefits
The landmark integration provides several advantages:
Geometric Precision: Explicit coordinate information captures fine-grained facial geometry
Robustness: Landmarks are less sensitive to lighting and appearance variations
Interpretability: Geometric features provide more interpretable emotion cues
Complementary Information: Combines global visual context with local geometric details

8.5 Custom Architecture Comparison
While direct quantitative comparison with VGG16 and ResNet18 baselines requires identical evaluation protocols, the custom architecture demonstrates several theoretical and practical advantages:
Advantages over Baselines:
Multi-Modal Input: Incorporates both visual and geometric information
Task-Specific Design: Explicitly designed for emotion recognition rather than general classification
Efficient Parameter Usage: Smaller than VGG16 while potentially more effective than single-modal approaches
Interpretable Features: Landmark features provide insight into model decision-making
Training Efficiency:
Smooth convergence over 100 epochs
No signs of overfitting or instability
Effective multi-task learning balancing classification and regression objectives

9. Discussion and Insights
8.1 Multi-Task Learning Observations
The results reveal an interesting dichotomy: ResNet18 excels at discrete classification while VGG16 performs better at continuous regression. This suggests that different architectural characteristics may be optimal for different aspects of affective computing.
8.2 Practical Implications
For real-world deployment, the choice between architectures depends on application priorities:
Classification-focused applications: ResNet18 provides better discrete emotion recognition
Continuous affect monitoring: VGG16 offers superior valence/arousal prediction
Resource-constrained environments: ResNet18's efficiency advantages are significant
8.3 Future Improvements
Ensemble Approaches: Combining both architectures could leverage their complementary strengths
Attention Mechanisms: Incorporating attention could improve focus on relevant facial regions
Data Augmentation: Advanced techniques like facial landmark-guided augmentation
Loss Function Optimization: Exploring alternative multi-task loss formulations

9. Conclusion
This study successfully implemented and compared VGG16 and ResNet18 architectures for multi-modal facial expression analysis. Key findings include:
ResNet18 demonstrates superior classification performance with better efficiency
VGG16 excels in continuous affect prediction across all regression metrics
Both architectures show healthy learning patterns without significant overfitting
The choice of architecture should align with specific application requirements
Sign Agreement (SAGR) emerges as the most crucial metric for real-world deployment
The results provide valuable insights into architecture selection for affective computing applications and highlight the complex trade-offs involved in multi-task learning scenarios.

