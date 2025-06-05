"""
🧠 QUOTEX AI PRICE ACTION TRAINER V2
===================================

Trains a regression model to predict price movements using OHLC data.
Matches the existing brain_latest.h5 model architecture.
"""

import os
import time
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    print("✅ TensorFlow and ML libraries loaded")
except ImportError as e:
    print(f"❌ Missing ML libraries: {e}")
    print("Install with: pip install tensorflow scikit-learn pandas numpy")
    exit(1)

# Quotex API
try:
    from quotexapi.stable_api import Quotex
    from quotexapi.expiration import get_timestamp_days_ago, timestamp_to_date
    from quotexapi.config import email, password
    print("✅ Quotex API loaded")
except ImportError as e:
    print(f"❌ Quotex API not found: {e}")
    exit(1)

# ============================================================================
# 🔧 CONFIGURATION
# ============================================================================

CONFIG = {
    "email": email,
    "password": password,
    "assets": {
        # Forex pairs (Quotex format)
        'EURUSD': 'EURUSD_otc',
        'GBPUSD': 'GBPUSD_otc',
        'USDJPY': 'USDJPY_otc',
        'AUDUSD': 'AUDUSD_otc',
        'USDCAD': 'USDCAD_otc',

        # Cryptocurrencies
        'BTCUSD': 'BTCUSD_otc',
        'ETHUSD': 'ETHUSD_otc',
        'XRPUSD': 'XRPUSD_otc',

        # Stocks
        'MSFT': 'MSFT_otc',
        'GOOGL': 'GOOGL_otc'
    },
    "period": 60,  # 1-minute candles
    "days": 20,  # 20 days of data per asset
    "sequence_length": 30,  # Match existing model
    "prediction_horizon": 1,  # Predict next 1 candle
    "model_path": "models/",
    "training_data_path": "training_data/",
    "raw_data_path": "training_data/raw/",
    "processed_data_path": "training_data/processed/",
    "features": ["open", "high", "low", "close"]  # Only OHLC features
}

# ============================================================================
# 📊 DATA FETCHER CLASS
# ============================================================================

class DataFetcher:

    def __init__(self):
        self.client = None

    async def connect(self, max_retries=3):
        """Connect to Quotex API with retry logic"""
        print("🔌 Connecting to Quotex...")

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}")

                self.client = Quotex(
                    email=CONFIG["email"],
                    password=CONFIG["password"],
                    lang="en"
                )

                self.client.debug_ws_enable = False

                check, reason = await self.client.connect()
                if check:
                    print("✅ Connected successfully!")

                    # Switch to demo account for safety
                    try:
                        if hasattr(self.client, 'change_account'):
                            change_result = self.client.change_account("PRACTICE")
                            if asyncio.iscoroutine(change_result):
                                await change_result
                            print("🎯 Switched to DEMO account")
                        else:
                            print("🎯 Using default account")
                    except Exception as e:
                        print(f"⚠️ Could not switch to demo account: {str(e)}")

                    return True
                else:
                    print(f"❌ Connection failed: {reason}")
                    if attempt < max_retries - 1:
                        print("🔄 Retrying in 5 seconds...")
                        await asyncio.sleep(5)

            except Exception as e:
                print(f"❌ Connection error: {e}")
                if attempt < max_retries - 1:
                    print("🔄 Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    print("❌ All connection attempts failed!")
                    return False

        return False

    async def fetch_historical_data(self, asset, asset_name, period, days):
        """Fetch historical data using chunks with progressive=True"""
        print(f"📈 Fetching {days} days data for {asset_name} ({asset})...")

        all_candles = []

        # Calculate chunk parameters for 60-second candles over 20 days
        # For 1-minute data, use 2-hour chunks (120 candles per chunk)
        chunk_hours = 2  # 2-hour chunks for 60-second candles
        chunk_offset = chunk_hours * 3600  # Convert to seconds (7200 seconds = 2 hours)
        total_chunks = (days * 24) // chunk_hours  # Total chunks needed (20 days = 240 chunks)

        # Calculate starting timestamp
        from quotexapi.expiration import get_timestamp_days_ago
        start_timestamp = get_timestamp_days_ago(days)
        end_from_time = (int(start_timestamp) - int(start_timestamp) % period) + chunk_offset

        expected_candles_per_chunk = chunk_hours * 60  # 120 candles per 2-hour chunk
        expected_total_candles = days * 24 * 60  # 28,800 candles for 20 days

        print(f"🔄 Fetching {total_chunks} chunks of {chunk_hours}h each for 60-second candles...")
        print(f"📊 Expected: {expected_candles_per_chunk} candles/chunk, {expected_total_candles:,} total candles")

        successful_chunks = 0
        failed_chunks = 0

        for chunk_num in range(total_chunks):
            try:
                # Progress indicator
                progress = (chunk_num + 1) / total_chunks * 100
                print(f"📊 {asset_name}: Chunk {chunk_num + 1}/{total_chunks} ({progress:.1f}%) - {len(all_candles):,} candles", end="\r")

                # Try multiple methods for each chunk
                chunk_candles = None

                # Method 1: Standard get_candles with progressive=True
                try:
                    chunk_candles = await asyncio.wait_for(
                        self.client.get_candles(
                            asset=asset,
                            end_from_time=end_from_time,
                            offset=chunk_offset,
                            period=period,
                            progressive=True
                        ),
                        timeout=10.0
                    )
                    if chunk_candles and len(chunk_candles) > 0:
                        all_candles.extend(chunk_candles)
                        successful_chunks += 1
                    else:
                        chunk_candles = None
                except Exception as e:
                    chunk_candles = None

                # Method 2: Try get_candle_v2 if first method fails
                if not chunk_candles:
                    try:
                        chunk_candles = await asyncio.wait_for(
                            self.client.get_candle_v2(asset, period),
                            timeout=10.0
                        )
                        if chunk_candles and len(chunk_candles) > 0:
                            # Filter candles for this time range
                            filtered_candles = [
                                c for c in chunk_candles
                                if end_from_time - chunk_offset <= c.get('time', 0) <= end_from_time
                            ]
                            if filtered_candles:
                                all_candles.extend(filtered_candles)
                                successful_chunks += 1
                        else:
                            chunk_candles = None
                    except Exception as e:
                        chunk_candles = None

                # Method 3: Try without _otc suffix
                if not chunk_candles and "_otc" in asset:
                    try:
                        clean_symbol = asset.replace("_otc", "")
                        chunk_candles = await asyncio.wait_for(
                            self.client.get_candles(
                                asset=clean_symbol,
                                end_from_time=end_from_time,
                                offset=chunk_offset,
                                period=period,
                                progressive=True
                            ),
                            timeout=10.0
                        )
                        if chunk_candles and len(chunk_candles) > 0:
                            all_candles.extend(chunk_candles)
                            successful_chunks += 1
                        else:
                            chunk_candles = None
                    except Exception as e:
                        chunk_candles = None

                if not chunk_candles:
                    failed_chunks += 1

                # Move to next chunk
                end_from_time += chunk_offset

                # Rate limiting - slower for 60-second data to avoid API limits
                await asyncio.sleep(0.2)

            except Exception as e:
                print(f"\n⚠️ Error in chunk {chunk_num + 1}: {e}")
                failed_chunks += 1
                continue

        print(f"\n📊 Chunk Summary for {asset_name}:")
        print(f"   ✅ Successful: {successful_chunks}/{total_chunks}")
        print(f"   ❌ Failed: {failed_chunks}/{total_chunks}")
        print(f"   📈 Total candles: {len(all_candles)}")

        if len(all_candles) == 0:
            print(f"❌ No data fetched for {asset_name}")
            return None

        # Remove duplicates and sort by time
        print(f"🔧 Removing duplicates and sorting...")
        unique_candles = []
        seen_times = set()

        for candle in all_candles:
            candle_time = candle.get('time', 0)
            if candle_time not in seen_times:
                seen_times.add(candle_time)
                unique_candles.append(candle)

        # Sort by time
        unique_candles.sort(key=lambda x: x.get('time', 0))

        print(f"✅ {asset_name}: {len(unique_candles)} unique candles after deduplication")

        # Validate data quality for 60-second candles
        expected_candles = days * 24 * 60  # 28,800 for 20 days
        coverage_percentage = (len(unique_candles) / expected_candles) * 100

        print(f"📊 {asset_name} Data Quality:")
        print(f"   📈 Collected: {len(unique_candles):,} candles")
        print(f"   🎯 Expected: {expected_candles:,} candles")
        print(f"   📊 Coverage: {coverage_percentage:.1f}%")

        if coverage_percentage < 50:
            print(f"⚠️ {asset_name}: Low data coverage ({coverage_percentage:.1f}%)")
        elif coverage_percentage >= 80:
            print(f"✅ {asset_name}: Excellent data coverage ({coverage_percentage:.1f}%)")
        else:
            print(f"👍 {asset_name}: Good data coverage ({coverage_percentage:.1f}%)")

        return unique_candles

    async def fetch_historical_data_fallback(self, asset, asset_name, period, days):
        """Fallback method using 1-hour chunks for 60-second candles"""
        print(f"🔄 Fallback: Fetching {days} days of 60-second data for {asset_name}...")

        all_candles = []
        offset = 3600  # 1 hour chunks (60 candles per chunk for 60-second data)
        size = days * 24  # Total hours needed

        from quotexapi.expiration import get_timestamp_days_ago
        start_timestamp = get_timestamp_days_ago(days)
        end_from_time = (int(start_timestamp) - int(start_timestamp) % period) + offset

        print(f"🔄 Fallback: {size} chunks of 1h each (60 candles/chunk)")

        successful_chunks = 0
        failed_chunks = 0

        for i in range(size):
            try:
                progress = (i + 1) / size * 100
                print(f"🔄 Fallback {asset_name}: {i+1}/{size} ({progress:.1f}%) - {len(all_candles):,} candles", end="\r")

                candles = await asyncio.wait_for(
                    self.client.get_candles(
                        asset=asset,
                        end_from_time=end_from_time,
                        offset=offset,
                        period=period,
                        progressive=True
                    ),
                    timeout=8.0
                )

                if candles and len(candles) > 0:
                    all_candles.extend(candles)
                    successful_chunks += 1
                else:
                    failed_chunks += 1

                end_from_time += offset
                await asyncio.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"\n⚠️ Fallback error in chunk {i+1}: {e}")
                failed_chunks += 1
                continue

        print(f"\n📊 Fallback Summary for {asset_name}:")
        print(f"   ✅ Successful: {successful_chunks}/{size}")
        print(f"   ❌ Failed: {failed_chunks}/{size}")
        print(f"   📈 Raw candles: {len(all_candles)}")

        # Remove duplicates and sort by time
        unique_candles = []
        seen_times = set()

        for candle in all_candles:
            candle_time = candle.get('time', 0)
            if candle_time not in seen_times:
                seen_times.add(candle_time)
                unique_candles.append(candle)

        unique_candles.sort(key=lambda x: x.get('time', 0))

        # Validate fallback data quality
        expected_candles = days * 24 * 60  # 28,800 for 20 days
        coverage = (len(unique_candles) / expected_candles) * 100

        print(f"✅ Fallback {asset_name}: {len(unique_candles):,} unique candles ({coverage:.1f}% coverage)")
        return unique_candles

    def convert_candles_to_dataframe(self, candles, asset_name):
        """Convert candle data to DataFrame (same logic as bot)"""
        try:
            df_data = []
            for candle in candles:
                try:
                    # Handle different candle data formats
                    if isinstance(candle, dict):
                        # Format 1: Direct dictionary with OHLC keys
                        if all(k in candle for k in ['open', 'high', 'low', 'close']):
                            df_data.append({
                                'open': float(candle['open']),
                                'high': float(candle['high']),
                                'low': float(candle['low']),
                                'close': float(candle['close']),
                                'time': candle.get('time', time.time())
                            })
                        # Format 2: Dictionary with different key names
                        elif all(k in candle for k in ['o', 'h', 'l', 'c']):
                            df_data.append({
                                'open': float(candle['o']),
                                'high': float(candle['h']),
                                'low': float(candle['l']),
                                'close': float(candle['c']),
                                'time': candle.get('t', candle.get('time', time.time()))
                            })
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 4:
                        # Format 3: Array format [time, open, close, high, low] or similar
                        if len(candle) >= 5:
                            df_data.append({
                                'open': float(candle[1]),
                                'high': float(candle[3]),
                                'low': float(candle[4]),
                                'close': float(candle[2]),
                                'time': candle[0]
                            })
                        else:
                            # Assume [open, high, low, close]
                            df_data.append({
                                'open': float(candle[0]),
                                'high': float(candle[1]),
                                'low': float(candle[2]),
                                'close': float(candle[3]),
                                'time': time.time()
                            })
                except (ValueError, TypeError, IndexError) as e:
                    print(f"⚠️ Skipping invalid candle data: {candle} - {str(e)}")
                    continue

            if len(df_data) < 50:
                print(f"⚠️ Insufficient data for {asset_name}: {len(df_data)} candles")
                return None

            # Create DataFrame
            data = pd.DataFrame(df_data)

            # Ensure proper data types
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # Remove any rows with NaN values
            data = data.dropna(subset=['open', 'high', 'low', 'close'])

            if len(data) < 50:
                print(f"⚠️ Too few valid rows after cleaning: {len(data)}")
                return None

            # Sort by time if time column exists
            if 'time' in data.columns:
                try:
                    data = data.sort_values('time').reset_index(drop=True)
                    # Convert time to datetime index
                    data.index = pd.to_datetime(data['time'], unit='s', errors='coerce')
                    data = data.drop('time', axis=1)
                except Exception as e:
                    print(f"⚠️ Could not process time column: {str(e)}")
                    data = data.reset_index(drop=True)

            # Keep only OHLC columns
            data = data[['open', 'high', 'low', 'close']].copy()

            print(f"✅ {asset_name}: Converted to DataFrame with {len(data)} rows")
            return data

        except Exception as e:
            print(f"❌ Error converting candles for {asset_name}: {str(e)}")
            return None

    def save_raw_data(self, candles, asset_name):
        """Save raw candle data to JSON file"""
        try:
            os.makedirs(CONFIG["raw_data_path"], exist_ok=True)

            filename = f"{asset_name}_raw_{CONFIG['days']}days.json"
            filepath = os.path.join(CONFIG["raw_data_path"], filename)

            # Save raw candles
            with open(filepath, 'w') as f:
                json.dump(candles, f, indent=2)

            # Save metadata
            metadata = {
                'asset': asset_name,
                'source': 'quotex_api',
                'period': CONFIG['period'],
                'days': CONFIG['days'],
                'fetch_time': datetime.now().isoformat(),
                'data_points': len(candles)
            }

            metadata_file = f"{asset_name}_raw_metadata.json"
            metadata_path = os.path.join(CONFIG["raw_data_path"], metadata_file)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"💾 {asset_name}: Raw data saved to {filepath}")

        except Exception as e:
            print(f"❌ Error saving raw data for {asset_name}: {str(e)}")

    def save_processed_data(self, data, asset_name):
        """Save processed DataFrame to CSV file"""
        try:
            os.makedirs(CONFIG["processed_data_path"], exist_ok=True)

            filename = f"{asset_name}_processed_{CONFIG['days']}days.csv"
            filepath = os.path.join(CONFIG["processed_data_path"], filename)

            # Save processed data
            data.to_csv(filepath, index=True)

            # Save processing metadata
            metadata = {
                'asset': asset_name,
                'processed': True,
                'processing_time': datetime.now().isoformat(),
                'data_points': len(data),
                'features': list(data.columns),
                'date_range': {
                    'start': data.index.min().isoformat() if hasattr(data.index.min(), 'isoformat') else str(data.index.min()),
                    'end': data.index.max().isoformat() if hasattr(data.index.max(), 'isoformat') else str(data.index.max())
                }
            }

            metadata_file = f"{asset_name}_processed_metadata.json"
            metadata_path = os.path.join(CONFIG["processed_data_path"], metadata_file)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"💾 {asset_name}: Processed data saved to {filepath}")

        except Exception as e:
            print(f"❌ Error saving processed data for {asset_name}: {str(e)}")

    async def close(self):
        if self.client:
            await self.client.close()

# ============================================================================
# 🤖 PRICE ACTION MODEL
# ============================================================================

class PriceActionModel:

    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns = CONFIG["features"]

    def prepare_data(self, df):
        """Prepare data for LSTM training"""
        print("📊 Preparing data for training...")

        # Use only OHLC features (same as existing model)
        features = df[self.feature_columns].values.astype(float)

        # Use the price_direction labels if available (from processed data)
        if 'price_direction' in df.columns:
            print("📈 Using price_direction labels from processed data")
            # Convert to one-hot encoding for classification
            targets = df['price_direction'].values

            # Remove last row if needed
            if len(features) > len(targets):
                features = features[:len(targets)]
            elif len(targets) > len(features):
                targets = targets[:len(features)]

        else:
            print("📈 Creating price movement targets")
            # Create target: next candle's price movement (regression)
            price_changes = df['close'].pct_change().shift(-1)  # Next candle's change

            # Convert to binary-like target (0-1 scale)
            targets = (price_changes + 1) / 2  # Normalize to 0-1 range
            targets = np.clip(targets, 0.1, 0.9)  # Clip extreme values

            # Remove last row (no future target)
            features = features[:-1]
            targets = targets[:-1].values

        # Remove NaN values
        valid_indices = ~np.isnan(targets) if targets.dtype == float else ~pd.isna(targets)
        features = features[valid_indices]
        targets = targets[valid_indices]

        # Scale features (important for LSTM)
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences for LSTM
        X_sequences, y_sequences = [], []

        for i in range(self.sequence_length, len(features_scaled)):
            X_sequences.append(features_scaled[i-self.sequence_length:i])
            y_sequences.append(targets[i])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        print(f"✅ Data shape: X={X_sequences.shape}, y={y_sequences.shape}")

        if y_sequences.dtype == float:
            print(f"📊 Target range: {y_sequences.min():.4f} to {y_sequences.max():.4f}")
        else:
            unique, counts = np.unique(y_sequences, return_counts=True)
            print(f"📊 Target distribution: {dict(zip(unique, counts))}")

        return X_sequences, y_sequences

    def build_model(self, input_shape, num_classes=3):
        """Build LSTM model matching existing brain_latest.h5 architecture"""
        print("🏗️ Building LSTM model...")
        print(f"📊 Input shape: {input_shape}")
        print(f"🎯 Output classes: {num_classes} (DOWN, NEUTRAL, UP)")

        model = Sequential([
            # First LSTM layer (match existing model)
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),

            # Second LSTM layer (match existing model)
            LSTM(32, return_sequences=False),
            Dropout(0.3),

            # Dense layer
            Dense(16, activation='relu'),
            Dropout(0.2),

            # Output layer (classification for 3 classes)
            Dense(num_classes, activation='softmax')  # Softmax for classification
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # For integer labels
            metrics=['accuracy']
        )

        self.model = model
        print("✅ Model built successfully!")
        print(f"📊 Model architecture matches brain_latest.h5")
        print(f"🔧 Total parameters: {model.count_params():,}")
        return model

    def train(self, X, y, validation_split=0.2, epochs=100):
        """Train the model"""
        print("🚀 Starting training...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
        ]

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        print("✅ Training completed!")
        return history

    def save_model(self, filepath):
        """Save model and metadata"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model
        self.model.save(filepath)

        # Save metadata
        metadata = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length
        }

        import pickle
        with open(filepath.replace('.h5', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        print(f"💾 Model saved to {filepath}")

# ============================================================================
# 🔧 HELPER FUNCTIONS
# ============================================================================

def add_training_features(df, asset_name):
    """Add technical features for training"""
    try:
        print(f"🔧 Adding training features for {asset_name}...")

        # Calculate additional features
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

        # Price labels for training (future price direction)
        df['future_price'] = df['close'].shift(-1)
        df['price_direction'] = np.where(
            df['future_price'] > df['close'] * 1.001, 2,  # UP (>0.1%)
            np.where(df['future_price'] < df['close'] * 0.999, 0, 1)  # DOWN (<-0.1%), NEUTRAL
        )

        # Remove NaN values
        df = df.dropna()

        print(f"✅ {asset_name}: Added features - {len(df)} valid points")
        return df

    except Exception as e:
        print(f"❌ Error adding features for {asset_name}: {str(e)}")
        return df

def save_collection_summary(successful_assets, failed_assets, all_processed_data):
    """Save collection summary to file"""
    try:
        summary_data = {
            'collection_time': datetime.now().isoformat(),
            'total_assets': len(CONFIG['assets']),
            'successful_assets': len(successful_assets),
            'failed_assets': len(failed_assets),
            'successful_list': successful_assets,
            'failed_list': failed_assets,
            'asset_summaries': {},
            'configuration': {
                'days': CONFIG['days'],
                'period': CONFIG['period'],
                'assets': CONFIG['assets']
            }
        }

        # Add individual asset summaries
        for asset in successful_assets:
            if asset in all_processed_data:
                data = all_processed_data[asset]
                summary_data['asset_summaries'][asset] = {
                    'raw_points': len(data),
                    'processed_points': len(data),
                    'date_range': {
                        'start': data.index.min().isoformat() if hasattr(data.index.min(), 'isoformat') else str(data.index.min()),
                        'end': data.index.max().isoformat() if hasattr(data.index.max(), 'isoformat') else str(data.index.max())
                    },
                    'label_distribution': {
                        'DOWN': int((data['price_direction'] == 0).sum()) if 'price_direction' in data.columns else 0,
                        'NEUTRAL': int((data['price_direction'] == 1).sum()) if 'price_direction' in data.columns else 0,
                        'UP': int((data['price_direction'] == 2).sum()) if 'price_direction' in data.columns else 0
                    }
                }

        summary_file = os.path.join(CONFIG["training_data_path"], "collection_summary.json")

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"💾 Collection summary saved to {summary_file}")

    except Exception as e:
        print(f"❌ Error saving summary: {str(e)}")

# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

async def main():
    """Main training pipeline for multiple assets"""
    print("🧠 QUOTEX AI PRICE ACTION TRAINER V2")
    print("=" * 50)
    print(f"🎯 Training with {len(CONFIG['assets'])} assets")
    print(f"📊 {CONFIG['days']} days of data per asset")
    print(f"💾 Storing raw + processed data")
    print("=" * 50)

    # Create directories
    os.makedirs(CONFIG["model_path"], exist_ok=True)
    os.makedirs(CONFIG["training_data_path"], exist_ok=True)
    os.makedirs(CONFIG["raw_data_path"], exist_ok=True)
    os.makedirs(CONFIG["processed_data_path"], exist_ok=True)

    # Initialize data fetcher
    fetcher = DataFetcher()

    # Storage for all processed data
    all_processed_data = {}
    successful_assets = []
    failed_assets = []

    try:
        # Connect to Quotex API
        if not await fetcher.connect():
            print("❌ Failed to connect to Quotex API")
            return

        # Process each asset
        for i, (asset_name, asset_symbol) in enumerate(CONFIG["assets"].items(), 1):
            print(f"\n📊 [{i}/{len(CONFIG['assets'])}] Processing {asset_name} ({asset_symbol})...")

            # Check if raw data already exists
            raw_data_file = os.path.join(CONFIG["raw_data_path"], f"{asset_name}_raw_{CONFIG['days']}days.json")
            processed_data_file = os.path.join(CONFIG["processed_data_path"], f"{asset_name}_processed_{CONFIG['days']}days.csv")

            # Load or fetch raw data
            if os.path.exists(raw_data_file):
                print(f"📁 Loading existing raw data for {asset_name}...")
                try:
                    with open(raw_data_file, 'r') as f:
                        raw_candles = json.load(f)
                    print(f"✅ Loaded {len(raw_candles)} existing candles")
                except Exception as e:
                    print(f"❌ Error loading existing data: {str(e)}")
                    raw_candles = None
            else:
                print(f"🔄 Fetching new data for {asset_name}...")
                raw_candles = await fetcher.fetch_historical_data(
                    asset_symbol, asset_name, CONFIG["period"], CONFIG["days"]
                )

                # If primary method didn't get enough data, try fallback
                # For 20 days of 60-second candles, expect at least 30% coverage (8,640 candles)
                expected_min_candles = CONFIG["days"] * 24 * 60 * 0.3  # 8,640 candles minimum
                if not raw_candles or len(raw_candles) < expected_min_candles:
                    print(f"🔄 Primary method insufficient, trying fallback for {asset_name}...")
                    raw_candles = await fetcher.fetch_historical_data_fallback(
                        asset_symbol, asset_name, CONFIG["period"], CONFIG["days"]
                    )

                if raw_candles and len(raw_candles) > 100:
                    # Save raw data
                    fetcher.save_raw_data(raw_candles, asset_name)
                    print(f"✅ {asset_name}: Saved {len(raw_candles)} candles")
                else:
                    print(f"❌ No sufficient data fetched for {asset_name}")
                    raw_candles = None

            # Process data if we have raw candles
            if raw_candles:
                # Convert to DataFrame
                df = fetcher.convert_candles_to_dataframe(raw_candles, asset_name)

                if df is not None and len(df) > 100:
                    # Add technical features for training
                    df = add_training_features(df, asset_name)

                    # Save processed data
                    fetcher.save_processed_data(df, asset_name)

                    # Store for training
                    all_processed_data[asset_name] = df
                    successful_assets.append(asset_name)

                    print(f"✅ {asset_name}: Successfully processed {len(df)} data points")
                else:
                    failed_assets.append(asset_name)
                    print(f"❌ {asset_name}: Failed to process data")
            else:
                failed_assets.append(asset_name)
                print(f"❌ {asset_name}: No raw data available")

            # Small delay between assets
            await asyncio.sleep(1)

    finally:
        # Always close the connection
        await fetcher.close()

    # Print collection summary
    print(f"\n📊 DATA COLLECTION SUMMARY:")
    print(f"✅ Successful: {len(successful_assets)} assets")
    print(f"❌ Failed: {len(failed_assets)} assets")

    if successful_assets:
        print(f"\n✅ Successful Assets:")
        for asset in successful_assets:
            data_points = len(all_processed_data[asset])
            print(f"   📈 {asset}: {data_points:,} data points")

    if failed_assets:
        print(f"\n❌ Failed Assets:")
        for asset in failed_assets:
            print(f"   💥 {asset}")

    # Save collection summary
    save_collection_summary(successful_assets, failed_assets, all_processed_data)

    # Train model if we have data
    if len(all_processed_data) > 0:
        print(f"\n🧠 TRAINING BRAIN MODEL...")
        print("=" * 50)

        # Combine all processed data for training
        combined_data = []
        for asset_name, df in all_processed_data.items():
            # Add asset identifier
            df_copy = df.copy()
            df_copy['asset'] = asset_name
            combined_data.append(df_copy)

        # Combine all DataFrames
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"📊 Combined training data: {len(combined_df):,} total data points")

        # Train the model
        model = PriceActionModel(CONFIG["sequence_length"])
        X, y = model.prepare_data(combined_df)

        if len(X) > 0:
            print(f"🎯 Training sequences: {len(X):,}")

            # Build and train model (3 classes: DOWN, NEUTRAL, UP)
            model.build_model((X.shape[1], X.shape[2]), num_classes=3)
            history = model.train(X, y)

            # Save model
            model_filename = f"{CONFIG['model_path']}/brain_latest.h5"
            model.save_model(model_filename)

            # Training Summary
            print("\n🎉 TRAINING COMPLETED!")
            print("=" * 50)
            print(f"📊 Assets used: {len(successful_assets)}")
            print(f"📈 Total data points: {len(combined_df):,}")
            print(f"🎯 Training sequences: {len(X):,}")
            print(f"🧠 Model saved: {model_filename}")

            if history and 'val_loss' in history.history:
                final_loss = min(history.history['val_loss'])
                print(f"📉 Best validation loss: {final_loss:.6f}")

            print(f"\n📁 DATA STORAGE:")
            print(f"   📂 Raw data: {CONFIG['raw_data_path']}")
            print(f"   🔧 Processed data: {CONFIG['processed_data_path']}")
            print(f"   📋 Summary: {os.path.join(CONFIG['training_data_path'], 'collection_summary.json')}")

        else:
            print("❌ No valid training sequences created!")
    else:
        print("\n❌ No data available for training!")
        print("🔍 Check your Quotex connection and asset availability")

    # 2. Process Data
    print("\n📊 Processing data...")

    # Handle different data formats
    if isinstance(raw_data, dict) and 'data' in raw_data:
        candle_data = raw_data['data']
    elif isinstance(raw_data, list):
        candle_data = raw_data
    else:
        print("❌ Unexpected data format!")
        return

    df = pd.DataFrame(candle_data)

    # Ensure numeric columns
    for col in CONFIG["features"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with NaN values
    df = df.dropna(subset=CONFIG["features"])
    df = df.sort_values('time').reset_index(drop=True)

    print(f"Data shape: {df.shape}")
    print(f"Features: {CONFIG['features']}")

    # 3. Train Model
    model = PriceActionModel(CONFIG["sequence_length"])
    X, y = model.prepare_data(df)

    if len(X) == 0:
        print("❌ No valid training data!")
        return

    # Build and train model
    model.build_model((X.shape[1], X.shape[2]))
    history = model.train(X, y)

    # 4. Save Model
    model_filename = f"{CONFIG['model_path']}/brain_latest_v2.h5"
    model.save_model(model_filename)

    # 5. Training Summary
    print("\n🎉 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Asset: {CONFIG['asset']}")
    print(f"Timeframe: {CONFIG['period']}s")
    print(f"Data points: {len(df)}")
    print(f"Training samples: {len(X)}")
    print(f"Features: {CONFIG['features']}")
    print(f"Model saved: {model_filename}")

    final_loss = min(history.history['val_loss'])
    print(f"Best validation loss: {final_loss:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
