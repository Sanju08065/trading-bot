"""
🧠 BRAIN MODEL - NEURAL NETWORK TRADING PREDICTOR 🧠
===================================================

Simple and fast neural network model for binary options trading.
Uses pre-trained brain_latest.h5 model for predictions.

Features:
- Fast neural network predictions
- Simple signal generation (UP/DOWN/NEUTRAL)
- Confidence scoring
- Optimal expiry time selection
- Minimal dependencies and overhead

Author: Trading Bot AI
Version: 1.0.0 - Brain Model Integration
License: MIT
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger.info("✅ TensorFlow loaded successfully")
except ImportError as e:
    logger.error(f"❌ TensorFlow not available: {e}")
    tf = None

try:
    from sklearn.preprocessing import MinMaxScaler
    logger.info("✅ Scikit-learn loaded successfully")
except ImportError as e:
    logger.error(f"❌ Scikit-learn not available: {e}")
    MinMaxScaler = None


class BrainModel:
    """
    🧠 BRAIN MODEL - Neural Network Trading Predictor

    Simple neural network model that loads pre-trained weights
    and provides fast trading predictions.
    """

    def __init__(self, model_path: str = "models/brain_model.h5"):
        """
        Initialize the Brain Model

        Args:
            model_path: Path to the pre-trained model file
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.sequence_length = 60  # Default sequence length

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the pre-trained neural network model"""
        try:
            if not tf:
                logger.error("❌ TensorFlow not available - cannot load brain model")
                return

            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                return

            # Load the model
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"✅ Brain model loaded successfully from {self.model_path}")

            # Initialize scaler (will be fitted on first use)
            if MinMaxScaler:
                self.scaler = MinMaxScaler()

            self.is_loaded = True

            # Log model info
            if hasattr(self.model, 'input_shape'):
                input_shape = self.model.input_shape
                logger.info(f"📊 Model input shape: {input_shape}")

                # Extract sequence length from input shape if available
                if len(input_shape) >= 2 and input_shape[1]:
                    self.sequence_length = input_shape[1]
                    logger.info(f"📏 Sequence length: {self.sequence_length}")

        except Exception as e:
            logger.error(f"❌ Failed to load brain model: {str(e)}")
            self.is_loaded = False

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLC data for the neural network

        Args:
            data: DataFrame with OHLC data

        Returns:
            Prepared feature array with 12 features to match model input shape (None, 30, 12)
        """
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()

            # Add the same technical features that were used during training
            # Calculate additional features (same as add_training_features in training script)
            df['price_change'] = df['close'].pct_change()
            df['hl_ratio'] = (df['high'] - df['low']) / df['close']
            df['oc_ratio'] = (df['close'] - df['open']) / df['open']

            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()

            # Price position relative to moving averages
            df['price_vs_sma5'] = df['close'] / df['sma_5']
            df['price_vs_sma10'] = df['close'] / df['sma_10']
            df['price_vs_sma20'] = df['close'] / df['sma_20']

            # Volatility
            df['volatility'] = df['price_change'].rolling(20).std()

            # Remove NaN values
            df = df.dropna()

            # Select the 12 features that the model expects (based on training script)
            # The model was trained with these 12 features:
            feature_columns = [
                'open', 'high', 'low', 'close',           # 4 OHLC features
                'price_change', 'hl_ratio', 'oc_ratio',   # 3 price ratios
                'price_vs_sma5', 'price_vs_sma10', 'price_vs_sma20',  # 3 SMA ratios
                'volatility',                              # 1 volatility feature
                'sma_5'                                   # 1 SMA value (to make 12 total)
            ]

            # Extract feature array
            feature_array = df[feature_columns].values.astype(float)

            # Handle any NaN values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)

            logger.info(f"🔧 Prepared features shape: {feature_array.shape} (12 features)")
            logger.debug(f"🔧 Feature columns: {feature_columns}")

            return feature_array

        except Exception as e:
            logger.error(f"❌ Error preparing features: {str(e)}")
            # Return basic OHLC as fallback with padding to 12 features
            try:
                basic_features = data[['open', 'high', 'low', 'close']].fillna(0).values
                # Pad with zeros to make 12 features
                padding = np.zeros((basic_features.shape[0], 8))
                return np.hstack([basic_features, padding])
            except:
                # Ultimate fallback
                return np.zeros((len(data), 12))

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using MinMaxScaler"""
        try:
            if not self.scaler or not MinMaxScaler:
                return features

            # Fit scaler on first use or if not fitted
            if not hasattr(self.scaler, 'scale_'):
                self.scaler.fit(features)

            return self.scaler.transform(features)

        except Exception as e:
            logger.warning(f"⚠️ Scaling failed, using raw features: {str(e)}")
            return features

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading prediction using the brain model

        Args:
            data: DataFrame with OHLC data (columns: open, high, low, close)

        Returns:
            dict: {
                'signal': 'UP', 'DOWN', or 'NEUTRAL',
                'confidence': float between 0.0 and 1.0,
                'details': dict with prediction details
            }
        """
        try:
            if not self.is_loaded:
                return self._fallback_prediction("Model not loaded")

            if len(data) < 10:
                return self._fallback_prediction("Insufficient data")

            # Prepare features
            features = self._prepare_features(data)

            # Scale features
            features_scaled = self._scale_features(features)

            # Prepare sequence for model
            if len(features_scaled) < self.sequence_length:
                # Pad with last values if not enough data
                padding_needed = self.sequence_length - len(features_scaled)
                last_row = features_scaled[-1:] if len(features_scaled) > 0 else np.zeros((1, features_scaled.shape[1]))
                padding = np.repeat(last_row, padding_needed, axis=0)
                features_scaled = np.vstack([padding, features_scaled])

            # Take last sequence_length rows
            sequence = features_scaled[-self.sequence_length:]

            # Reshape for model input (batch_size, sequence_length, features)
            model_input = sequence.reshape(1, self.sequence_length, -1)

            # Make prediction
            prediction = self.model.predict(model_input, verbose=0)[0]

            # Convert prediction to signal
            signal, confidence = self._interpret_prediction(prediction)

            # Determine optimal expiry
            optimal_expiry = self._get_optimal_expiry(signal, confidence, data)

            return {
                'signal': signal,
                'confidence': confidence,
                'details': {
                    'model_type': 'brain_neural_network',
                    'prediction_raw': prediction.tolist() if hasattr(prediction, 'tolist') else str(prediction),
                    'optimal_expiry': optimal_expiry,
                    'data_points_used': len(data),
                    'sequence_length': self.sequence_length,
                    'features_count': features_scaled.shape[1] if len(features_scaled.shape) > 1 else 1
                }
            }

        except Exception as e:
            logger.error(f"❌ Brain model prediction failed: {str(e)}")
            return self._fallback_prediction(f"Prediction error: {str(e)}")

    def _interpret_prediction(self, prediction) -> Tuple[str, float]:
        """
        Interpret raw model prediction into signal and confidence

        Args:
            prediction: Raw model output

        Returns:
            Tuple of (signal, confidence)
        """
        try:
            # Handle different prediction formats
            if hasattr(prediction, 'shape') and len(prediction.shape) > 0:
                if len(prediction) == 1:
                    # Single output (regression-like)
                    value = float(prediction[0])
                    if value > 0.6:
                        return 'UP', min(value, 0.95)
                    elif value < 0.4:
                        return 'DOWN', min(1.0 - value, 0.95)
                    else:
                        return 'NEUTRAL', 0.1

                elif len(prediction) == 2:
                    # Binary classification [down_prob, up_prob]
                    down_prob, up_prob = float(prediction[0]), float(prediction[1])
                    if up_prob > down_prob and up_prob > 0.6:
                        return 'UP', min(up_prob, 0.95)
                    elif down_prob > up_prob and down_prob > 0.6:
                        return 'DOWN', min(down_prob, 0.95)
                    else:
                        return 'NEUTRAL', 0.1

                elif len(prediction) == 3:
                    # Three-class [down, neutral, up]
                    down_prob, neutral_prob, up_prob = prediction[0], prediction[1], prediction[2]
                    max_prob = max(down_prob, neutral_prob, up_prob)

                    if max_prob == up_prob and up_prob > 0.5:
                        return 'UP', min(float(up_prob), 0.95)
                    elif max_prob == down_prob and down_prob > 0.5:
                        return 'DOWN', min(float(down_prob), 0.95)
                    else:
                        return 'NEUTRAL', 0.1

                else:
                    # Multi-class, take argmax
                    predicted_class = np.argmax(prediction)
                    confidence = float(np.max(prediction))

                    if predicted_class == 0:
                        return 'DOWN', min(confidence, 0.95)
                    elif predicted_class == 1:
                        return 'NEUTRAL', 0.1
                    else:
                        return 'UP', min(confidence, 0.95)
            else:
                # Scalar prediction
                value = float(prediction)
                if value > 0.6:
                    return 'UP', min(value, 0.95)
                elif value < 0.4:
                    return 'DOWN', min(1.0 - value, 0.95)
                else:
                    return 'NEUTRAL', 0.1

        except Exception as e:
            logger.error(f"❌ Error interpreting prediction: {str(e)}")
            return 'NEUTRAL', 0.1

    def _get_optimal_expiry(self, signal: str, confidence: float, data: pd.DataFrame) -> int:
        """
        Determine optimal expiry time based on signal and market conditions

        Args:
            signal: Trading signal
            confidence: Prediction confidence
            data: Market data

        Returns:
            Optimal expiry time in seconds
        """
        try:
            # Base expiry times (API supported)
            base_expiries = [60, 90]  # 60s and 90s only as per user preference

            if signal == 'NEUTRAL':
                return 60  # Default for neutral signals

            # High confidence gets shorter expiry for quick profits
            if confidence >= 0.8:
                return 60
            elif confidence >= 0.6:
                return 90
            else:
                return 60  # Default fallback

        except Exception as e:
            logger.error(f"❌ Error calculating optimal expiry: {str(e)}")
            return 60  # Safe fallback

    def _fallback_prediction(self, reason: str) -> Dict[str, Any]:
        """Return a safe fallback prediction when model fails"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.0,
            'details': {
                'model_type': 'brain_neural_network',
                'error': reason,
                'optimal_expiry': 60,
                'fallback': True
            }
        }


# Global brain model instance
_brain_model_instance = None

def get_brain_model() -> BrainModel:
    """Get or create global brain model instance"""
    global _brain_model_instance
    if _brain_model_instance is None:
        _brain_model_instance = BrainModel()
    return _brain_model_instance

def get_brain_prediction(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Simple function interface for brain model predictions

    Args:
        data: DataFrame with OHLC data

    Returns:
        Prediction dictionary
    """
    brain_model = get_brain_model()
    return brain_model.predict(data)


if __name__ == "__main__":
    """Test the brain model with sample data"""
    print("🧠 Testing Brain Model...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(1.1000, 1.1100, 100),
        'high': np.random.uniform(1.1050, 1.1150, 100),
        'low': np.random.uniform(1.0950, 1.1050, 100),
        'close': np.random.uniform(1.1000, 1.1100, 100),
    }, index=dates)

    # Test prediction
    brain_model = BrainModel()
    result = brain_model.predict(sample_data)

    print(f"📊 Test Result:")
    print(f"   Signal: {result['signal']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Optimal Expiry: {result['details']['optimal_expiry']}s")
    print(f"   Model Loaded: {brain_model.is_loaded}")
