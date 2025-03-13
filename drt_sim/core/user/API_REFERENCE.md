# API Reference

## Feature Extraction System

### FeatureExtractor

```python
class FeatureExtractor:
    def __init__(
        self,
        feature_registry=None,
        enabled_features=None,
        normalization_overrides=None
    )
```

**Parameters**:
- `feature_registry`: Optional dictionary containing feature metadata
- `enabled_features`: Optional list of features to enable (None = all)
- `normalization_overrides`: Optional dictionary of normalization values

**Methods**:

```python
def get_feature_dim(self) -> int
```
Returns the dimensionality of the feature vector.

```python
def get_feature_names(self) -> List[str]
```
Returns the names of enabled features.

```python
def extract_features_dict(
    self,
    features: Dict[str, Any],
    request: Optional[Request] = None,
    user_profile: Optional[UserProfile] = None
) -> Dict[str, float]
```
Extracts normalized features as a dictionary.

```python
def extract_features_vector(
    self,
    features: Dict[str, Any],
    request: Optional[Request] = None,
    user_profile: Optional[UserProfile] = None
) -> np.ndarray
```
Extracts normalized features as a numpy vector.

```python
def get_feature_groups(self) -> Dict[str, List[str]]
```
Gets features grouped by their categories.

```python
def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]
```
Gets metadata for a specific feature.

```python
def add_custom_feature(
    self,
    name: str,
    description: str,
    unit: str,
    normalization: float,
    importance: str = "medium",
    group: str = "custom"
) -> None
```
Adds a custom feature to the registry.

### FeatureProvider (Base Class)

```python
class FeatureProvider(abc.ABC):
    @abc.abstractmethod
    def get_features(
        self,
        request: Optional[Request],
        context: Dict[str, Any]
    ) -> Dict[str, Any]
```
Abstract method that returns a dictionary of features.

```python
def get_feature_names(self) -> List[str]
```
Returns the names of features provided by this provider.

### FeatureProviderRegistry

```python
class FeatureProviderRegistry:
    def __init__(self)
```
Creates a new feature provider registry.

**Methods**:

```python
def register_provider(self, name: str, provider: FeatureProvider) -> None
```
Registers a feature provider.

```python
def get_features(
    self,
    request: Optional[Request],
    context: Dict[str, Any],
    provider_names: Optional[List[str]] = None
) -> Dict[str, Any]
```
Gets features from specified providers (or all if provider_names is None).

```python
def get_all_feature_names(self) -> List[str]
```
Gets names of all available features across all providers.

```python
def get_provider_names(self) -> List[str]
```
Gets names of all registered providers.

## User Profile System

### UserProfile

```python
@dataclass
class UserProfile:
    id: str
    max_waiting_time: float = 15.0
    max_travel_time: float = 45.0
    max_cost: float = 30.0
    max_detour_ratio: float = 0.5
    service_preference: ServicePreference = ServicePreference.SPEED
    weights: Dict[str, float] = field(default_factory=lambda: {...})
    historical_trips: int = 0
    historical_acceptance_rate: float = 0.0
    historical_ratings: List[float] = field(default_factory=list)
    _manager: Any = None
```

**Methods**:

```python
def get_acceptance_rate(self) -> float
```
Gets the historical acceptance rate for this user.

```python
def get_trip_count(self) -> int
```
Gets the number of trips taken by this user.

```python
def get_average_rating(self) -> float
```
Gets the average rating given by this user.

```python
def update_acceptance_rate(self, accepted: bool) -> None
```
Updates the historical acceptance rate with a new decision.

```python
def add_rating(self, rating: float) -> None
```
Adds a new rating given by this user.

```python
def get_weights(self) -> Dict[str, float]
```
Gets weights for acceptance model features.

```python
def update_weights(self, new_weights: Dict[str, float]) -> None
```
Updates feature weights.

```python
def to_dict(self) -> Dict[str, Any]
```
Converts to dictionary for serialization.

```python
@classmethod
def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile'
```
Creates from dictionary representation.

### UserProfileManager

```python
class UserProfileManager:
    def __init__(self, profiles_dir: str = "user_profiles")
```
Initializes the user profile manager with a directory for storing profiles.

**Methods**:

```python
def get_user_profile(self, user_id: str) -> Optional[UserProfile]
```
Gets a user profile by ID, or None if not found.

```python
def get_or_create_user_profile(self, user_id: str) -> UserProfile
```
Gets a user profile, creating a new one if it doesn't exist.

```python
def save_user_profile(self, profile: UserProfile) -> None
```
Saves a user profile.

```python
def get_user_weights(self, user_id: str) -> Dict[str, float]
```
Gets weights for a specific user.

```python
def update_user_weights(self, user_id: str, new_weights: Dict[str, float]) -> None
```
Updates weights for a specific user.

```python
def get_all_user_ids(self) -> List[str]
```
Gets all user IDs.

```python
def delete_user_profile(self, user_id: str) -> bool
```
Deletes a user profile, returning True if successful.

## Acceptance Context

```python
class AcceptanceContext:
    def __init__(
        self,
        request: Request,
        features: Dict[str, Any],
        user_profile: Optional[UserProfile] = None
    )
```
Initializes an acceptance context with a request, features, and optional user profile.

**Methods**:

```python
def add_additional_data(self, key: str, value: Any) -> None
```
Adds additional data to the context.

```python
def get_additional_data(self, key: str, default: Any = None) -> Any
```
Gets additional data from the context.

```python
def update_features(self, new_features: Dict[str, Any]) -> None
```
Updates features in the context.

```python
def has_feature(self, feature_name: str) -> bool
```
Checks if a feature is present in the context.

```python
def get_feature(self, feature_name: str, default: Any = None) -> Any
```
Gets a feature value from the context.

```python
def to_dict(self) -> Dict[str, Any]
```
Converts the context to a dictionary.

```python
@classmethod
def from_dict(cls, data: Dict[str, Any], request: Request, user_profile: Optional[UserProfile] = None) -> 'AcceptanceContext'
```
Creates a context from a dictionary.

## User Acceptance Models

### UserAcceptanceModel (Base Class)

```python
class UserAcceptanceModel(abc.ABC):
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, **kwargs)
```
Initializes the base user acceptance model with an optional feature extractor.

**Methods**:

```python
@abc.abstractmethod
def calculate_acceptance_probability(self, context: AcceptanceContext) -> float
```
Calculates probability of user accepting a proposed service.

```python
def decide_acceptance(self, context: AcceptanceContext) -> Tuple[bool, float]
```
Decides whether the user will accept the proposed service.

```python
@abc.abstractmethod
def update_model(self, context: AcceptanceContext, accepted: bool) -> None
```
Updates the model based on user decisions.

```python
def batch_update(self, training_data: List[Dict[str, Any]]) -> None
```
Updates the model with batch training data.

```python
def get_feature_importance(self) -> Dict[str, float]
```
Gets the importance of each feature in the model.

```python
def get_required_features(self) -> List[str]
```
Gets the list of required features for this model.

```python
def get_optional_features(self) -> List[str]
```
Gets the list of optional features for this model.

```python
def configure(self, config: Dict[str, Any]) -> None
```
Configures the model with the given configuration.

### DefaultModel

```python
class DefaultModel(UserAcceptanceModel):
    def __init__(
        self,
        max_waiting_time: float = 15.0,
        max_travel_time: float = 45.0,
        max_cost: float = 30.0,
        feature_weights: Optional[Dict[str, float]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        **kwargs
    )
```
Initializes the default user acceptance model with threshold parameters.

### LogitModel

```python
class LogitModel(UserAcceptanceModel):
    def __init__(
        self,
        beta: float = 1.0,
        learning_rate: float = 0.01,
        feature_extractor: Optional[FeatureExtractor] = None,
        initial_coefficients: Optional[Dict[str, float]] = None,
        **kwargs
    )
```
Initializes the logit-based user acceptance model with given parameters.

**Additional Methods**:

```python
def save_model(self, path: str) -> None
```
Saves the model to a file.

```python
def load_model(self, path: str) -> None
```
Loads the model from a file.

```python
def get_convergence_metrics(self) -> Dict[str, Any]
```
Gets metrics about the model's training convergence.

### RLAcceptanceModel

```python
class RLAcceptanceModel(UserAcceptanceModel):
    def __init__(
        self,
        alpha: float = 0.001,
        beta: float = 1.0,
        gamma: float = 0.95,
        epsilon: float = 0.05,
        feature_extractor: Optional[FeatureExtractor] = None,
        user_models_dir: Optional[str] = None,
        **kwargs
    )
```
Initializes the RL-based user acceptance model with given parameters.

**Additional Methods**:

```python
def save_model(self, path: str) -> None
```
Saves the model to a file.

```python
def load_model(self, path: str) -> None
```
Loads the model from a file.

### ModelFactory

```python
class ModelFactory:
    @classmethod
    def register_model_type(cls, name: str, model_class: Type[UserAcceptanceModel]) -> None
```
Registers a new model type.

```python
@classmethod
def create_model(
    cls,
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    feature_extractor: Optional[FeatureExtractor] = None,
    feature_provider_registry: Optional[FeatureProviderRegistry] = None,
    **kwargs
) -> UserAcceptanceModel
```
Creates a user acceptance model of the specified type.

```python
@classmethod
def get_available_model_types(cls) -> List[str]
```
Gets the list of available model types.

```python
@classmethod
def get_model_class(cls, model_type: str) -> Type[UserAcceptanceModel]
```
Gets the class for a model type.