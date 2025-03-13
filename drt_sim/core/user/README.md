# User Acceptance Models for DRT Systems

This library provides a comprehensive framework for modeling user acceptance in Demand-Responsive Transit (DRT) systems. It allows you to predict whether users will accept proposed transportation services based on various service attributes and user preferences.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
  - [Feature Extraction](#feature-extraction)
  - [User Profiles](#user-profiles)
  - [Acceptance Context](#acceptance-context)
  - [Model Types](#model-types)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Extending the Framework](#extending-the-framework)
  - [Adding New Features](#adding-new-features)
  - [Creating Custom Models](#creating-custom-models)
- [API Reference](#api-reference)

## Overview

User acceptance modeling is critical for DRT systems to anticipate user behavior and optimize service delivery. This library provides a flexible, feature-based approach that can be tailored to different service contexts and user segments.

Key capabilities:
- Predict acceptance probabilities for proposed services
- Model different user preferences and decision factors
- Learn from past user decisions through various modeling approaches
- Support personalization at the individual user level

## Architecture

The system follows a modular design with clear separation of concerns:

![Architecture Diagram](./architecture.md)

The data flow works as follows:

1. Raw features are collected from various sources
2. Feature providers enrich the feature set
3. The feature extractor normalizes and prepares features
4. The acceptance context encapsulates all relevant information
5. Models process the context to produce acceptance probabilities
6. User-specific data influences the decision process
7. Model updates incorporate feedback from actual user decisions

## Core Components

### Feature Extraction

The feature extraction subsystem is responsible for collecting, normalizing, and organizing the features that influence acceptance decisions.

#### FeatureExtractor

The `FeatureExtractor` handles normalization and organization of features:

```python
# Create a feature extractor with custom normalization
feature_extractor = FeatureExtractor(
    normalization_overrides={
        "waiting_time": 20.0,  # 20 minutes is normalized to 1.0
        "travel_time": 40.0,   # 40 minutes is normalized to 1.0
        "cost": 25.0           # $25 is normalized to 1.0
    }
)

# Extract features as a dictionary
features_dict = feature_extractor.extract_features_dict(
    features=raw_features,
    request=request,
    user_profile=user_profile
)

# Extract features as a vector
feature_vector = feature_extractor.extract_features_vector(
    features=raw_features,
    request=request,
    user_profile=user_profile
)
```

#### Feature Providers

Feature providers enrich the feature set with specific categories of information:

```python
# Create a feature provider registry
registry = FeatureProviderRegistry()

# Register various providers
registry.register_provider("time", TimeBasedFeatureProvider())
registry.register_provider("spatial", SpatialFeatureProvider())
registry.register_provider("user_history", UserHistoryFeatureProvider())

# Get features from all providers
enriched_features = registry.get_features(
    request=request,
    context={"user_profile": user_profile, "current_time": datetime.now()}
)
```

Available providers:
- `TimeBasedFeatureProvider`: Time of day, day of week, etc.
- `SpatialFeatureProvider`: Location-based features
- `UserHistoryFeatureProvider`: User historical behavior
- `ServiceQualityFeatureProvider`: Vehicle and driver attributes
- `WeatherFeatureProvider`: Weather conditions (if weather service available)

### User Profiles

The user profile system manages user-specific data and preferences that influence acceptance behavior.

#### UserProfile

Each user is represented by a `UserProfile` with preferences and historical data:

```python
# Create a user profile
profile = UserProfile(
    id="user123",
    max_waiting_time=15.0,
    max_travel_time=45.0,
    service_preference=ServicePreference.SPEED
)

# Get user-specific weights
weights = profile.get_weights()

# Update user data based on decisions
profile.update_acceptance_rate(accepted=True)
```

#### UserProfileManager

The `UserProfileManager` handles loading, saving, and retrieving user profiles:

```python
# Create a profile manager
manager = UserProfileManager("data/user_profiles")

# Get a user profile (creates if not exists)
profile = manager.get_or_create_user_profile("user123")

# Get user-specific weights
weights = manager.get_user_weights("user123")

# Update user weights
manager.update_user_weights("user123", {"waiting_time": 0.5, "cost": 0.3})
```

### Acceptance Context

The `AcceptanceContext` encapsulates all the information needed for acceptance decisions:

```python
# Create a context
context = AcceptanceContext(
    request=request,
    features=features,
    user_profile=user_profile
)

# Access context data
waiting_time = context.get_feature("waiting_time")

# Add additional information
context.add_additional_data("weather_condition", "rainy")
```

### Model Types

The framework includes several model types for acceptance prediction:

#### DefaultModel

A simple threshold-based model:

```python
model = DefaultModel(
    max_waiting_time=15.0,
    max_travel_time=45.0,
    max_cost=30.0,
    feature_weights={
        "waiting_time": 0.4,
        "travel_time": 0.3,
        "cost": 0.3
    }
)
```

#### LogitModel

A logistic regression model for more complex decision boundaries:

```python
model = LogitModel(
    beta=1.0,
    learning_rate=0.01,
    initial_coefficients={
        "waiting_time": -0.5,
        "travel_time": -0.3,
        "cost": -0.2
    }
)
```

#### RLAcceptanceModel

A reinforcement learning model that adapts to changing preferences:

```python
model = RLAcceptanceModel(
    alpha=0.001,
    beta=1.0,
    gamma=0.95,
    epsilon=0.05,
    user_models_dir="data/user_models"
)
```

#### ModelFactory

The `ModelFactory` simplifies model creation and configuration:

```python
# Create a model using the factory
model = ModelFactory.create_model(
    model_type="logit",
    feature_extractor=feature_extractor,
    beta=2.0,
    learning_rate=0.02
)

# Get available model types
model_types = ModelFactory.get_available_model_types()
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/drt-sim.git
cd drt-sim

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from drt_sim.algorithms.user_acceptance_models.model_factory import ModelFactory
from drt_sim.algorithms.user_acceptance_models.acceptance_context import AcceptanceContext
from drt_sim.models.user_profile import UserProfileManager

# Create a user profile manager
profile_manager = UserProfileManager("data/user_profiles")

# Create a model
model = ModelFactory.create_model("default")

# Prepare features
features = {
    "waiting_time": 10.0,
    "travel_time": 20.0,
    "cost": 15.0,
    "detour_ratio": 0.2
}

# Get a user profile
user_profile = profile_manager.get_or_create_user_profile("user123")

# Create context
context = AcceptanceContext(
    request=request,
    features=features,
    user_profile=user_profile
)

# Calculate acceptance probability
probability = model.calculate_acceptance_probability(context)

# Make decision
accepted, probability = model.decide_acceptance(context)

# Update model based on actual user decision
model.update_model(context, accepted=True)
```

## Usage Examples

### Complete Simulation Example

The `complete_example.py` script demonstrates a full simulation with multiple users and model types:

```bash
python examples/complete_example.py
```

This will:
1. Set up the acceptance modeling system
2. Create multiple user profiles with different preferences
3. Generate random transportation requests
4. Simulate user decisions
5. Train and evaluate different model types
6. Visualize the results

### Comparing Model Performance

```python
# Create different model types
models = {
    "default": ModelFactory.create_model("default"),
    "logit": ModelFactory.create_model("logit"),
    "rl": ModelFactory.create_model("rl")
}

# Run evaluation
results = {}
for name, model in models.items():
    correct = 0
    total = 0
    
    for features, actual_accepted in test_data:
        context = AcceptanceContext(
            request=request,
            features=features,
            user_profile=user_profile
        )
        predicted_accepted, _ = model.decide_acceptance(context)
        correct += 1 if predicted_accepted == actual_accepted else 0
        total += 1
    
    results[name] = correct / total

print(results)
```

## Extending the Framework

### Adding New Features

To add a new feature to the system:

1. Add feature metadata to the `FeatureExtractor`:

```python
feature_extractor.add_custom_feature(
    name="traffic_congestion",
    description="Traffic congestion level",
    unit="level",
    normalization=1.0,
    importance="medium",
    group="environmental"
)
```

2. Create a custom feature provider if needed:

```python
class TrafficFeatureProvider(FeatureProvider):
    def get_features(self, request, context):
        return {
            "traffic_congestion": self.traffic_service.get_congestion(
                request.origin_location, 
                context.get("proposed_pickup_time")
            )
        }
    
    def get_feature_names(self):
        return ["traffic_congestion"]

# Register the provider
registry.register_provider("traffic", TrafficFeatureProvider())
```

### Creating Custom Models

To create a custom acceptance model:

1. Inherit from the base class:

```python
from drt_sim.algorithms.user_acceptance_models.user_acceptance_base import UserAcceptanceModel

class MyCustomModel(UserAcceptanceModel):
    def __init__(self, feature_extractor=None, **kwargs):
        super().__init__(feature_extractor=feature_extractor, **kwargs)
        # Initialize your model
        
    def calculate_acceptance_probability(self, context):
        # Extract features
        features = self.feature_extractor.extract_features_dict(
            features=context.features,
            request=context.request,
            user_profile=context.user_profile
        )
        
        # Your probability calculation logic
        return probability
        
    def update_model(self, context, accepted):
        # Your model update logic
        pass
```

2. Register with the factory:

```python
from drt_sim.algorithms.user_acceptance_models.model_factory import ModelFactory

ModelFactory.register_model_type("my_custom", MyCustomModel)
```

3. Use it like any other model:

```python
model = ModelFactory.create_model("my_custom")
```

## API Reference

### Feature System

- `FeatureExtractor`: Normalizes and processes features
- `FeatureProviderRegistry`: Manages feature providers
- `FeatureProvider`: Base class for feature providers

### User Profiles

- `UserProfile`: Stores user preferences and historical data
- `UserProfileManager`: Manages user profiles
- `ServicePreference`: Enum of user service preferences

### Context

- `AcceptanceContext`: Encapsulates request, features, and user profile

### Models

- `UserAcceptanceModel`: Base class for all models
- `DefaultModel`: Simple threshold-based model
- `LogitModel`: Logistic regression model
- `RLAcceptanceModel`: Reinforcement learning model
- `ModelFactory`: Factory for creating models

For detailed API documentation, see the [API Reference](./API_REFERENCE.md).