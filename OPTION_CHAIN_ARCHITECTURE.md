# Option Chain Architecture Documentation

## Overview

The option chain system has been completely redesigned to follow Kite Connect's recommended patterns and best practices. The new architecture provides better performance, reliability, and maintainability while maintaining full backward compatibility.

## Architecture Components

### 1. OptionChainBuilder (`src/provider/option_chain_builder.py`)

The main orchestrator that coordinates all option chain building operations.

**Key Features:**
- **Instruments Filtering**: Efficiently filters Kite instruments by symbol/expiry/strike range
- **Batch Quote Requests**: Handles quote requests in chunks ≤500 tokens with rate limiting
- **Data Assembly**: Assembles complete option chains from instruments and quotes
- **Performance Optimization**: Caching, strike limiting, and memory optimization
- **Backward Compatibility**: Converts to legacy format for existing code

**Usage:**
```python
from src.provider.kite import KiteProvider

provider = KiteProvider()
# provider.get_option_chain() now uses OptionChainBuilder internally
chain = provider.get_option_chain("NIFTY", "2024-01-01")
```

### 2. InstrumentFilter

Handles filtering of Kite instruments data.

**Capabilities:**
- Filter by symbol, exchange (NFO/BFO), expiry date, strike range
- Automatic exchange selection (NFO for most indices, BFO for SENSEX)
- Strike step inference from available strikes
- Vectorized pandas operations for performance

### 3. QuoteBatcher  

Manages efficient batch quote requests with rate limiting.

**Features:**
- Batch size optimization (≤500 tokens per request)
- Rate limiting between batches (200ms default)
- Error resilience (continues on partial failures)
- Comprehensive logging

### 4. OptionDataAssembler

Assembles final option chain structure from raw data.

**Functions:**
- Parse quote data into standardized format
- Handle missing/invalid data gracefully
- Calculate data quality scores
- Build structured OptionChain objects

### 5. Data Structures

**OptionData**: Type-safe option data for individual strikes
```python
@dataclass
class OptionData:
    strike: int
    option_type: str  # "CE" or "PE" 
    bid: float
    ask: float
    ltp: float
    bid_qty: int
    ask_qty: int
    oi: int
    volume: int
    # ... etc
```

**OptionChain**: Complete option chain with metadata
```python
@dataclass  
class OptionChain:
    symbol: str
    expiry: str
    spot_price: float
    strikes: List[int]
    options: Dict[Tuple[int, str], OptionData]  # (strike, type) -> data
    build_timestamp: dt.datetime
    data_quality_score: float  # 0.0 to 1.0
```

## Performance Optimizations

### 1. Caching
- **Expiry data caching**: 5-minute TTL for expiry lists
- **Instruments caching**: Reused across multiple option chain builds

### 2. Strike Limiting
- **Automatic limiting**: Limits to 100 strikes maximum if too many available
- **Distance-based selection**: Keeps strikes closest to spot price
- **Range-based filtering**: ±20% around spot by default

### 3. Memory Optimization
- **Vectorized operations**: Uses pandas `.loc` for efficient filtering
- **Batch processing**: Processes quotes in optimal batch sizes
- **Lazy loading**: OptionChainBuilder instantiated only when needed

## Backward Compatibility

The new system maintains 100% backward compatibility:

```python
# Existing code continues to work unchanged
provider = KiteProvider()
chain = provider.get_option_chain("NIFTY", "2024-01-01")

# Legacy format is preserved:
# chain = {
#     "symbol": "NIFTY",
#     "expiry": "2024-01-01", 
#     "strikes": [24000, 24050, ...],
#     "calls": {24000: {"ltp": 100.0, "oi": 1000, ...}, ...},
#     "puts": {24000: {"ltp": 120.0, "oi": 800, ...}, ...}
# }
```

## Error Handling

### 1. Graceful Degradation
- **Empty chains**: Returns empty structure instead of crashing
- **Partial data**: Continues with available data, reports quality score
- **API failures**: Logs errors and attempts to continue

### 2. Data Quality Metrics
- **Quality scores**: 0.0 to 1.0 based on successful quote retrieval
- **Comprehensive logging**: Detailed logs for debugging
- **Performance metrics**: Build times and data point counts

## Migration Guide

### For Existing Code
No changes required - the new system is drop-in compatible.

### For New Development
Use the new structures for better type safety and performance:

```python
from src.provider.option_chain_builder import OptionChainBuilder

# Direct usage (advanced)
builder = OptionChainBuilder(kite_client, instruments_df)
option_chain = builder.build_chain("NIFTY", "2024-01-01", 24000.0)

# Access structured data
for (strike, option_type), option_data in option_chain.options.items():
    print(f"{strike} {option_type}: LTP={option_data.ltp}, OI={option_data.oi}")
```

## Benefits of New Architecture

### 1. **Accuracy**
- Follows Kite Connect recommended patterns
- Proper instruments filtering and quote batching
- Better error handling and data validation

### 2. **Performance** 
- Optimized batch sizes and rate limiting
- Caching for frequently accessed data
- Memory-efficient pandas operations

### 3. **Reliability**
- Graceful handling of API failures
- Comprehensive error logging
- Data quality scoring

### 4. **Maintainability**
- Clean separation of concerns
- Type-safe data structures
- Comprehensive documentation and tests

### 5. **Extensibility**
- Modular design allows easy enhancements
- Support for new indices and markets
- Configurable performance parameters

## Testing

Comprehensive test suite covers:
- Individual component functionality
- Integration testing
- Performance validation
- Error condition handling
- Backward compatibility verification

Run tests with:
```bash
python test_option_chain_integration.py
```

## Legacy Code

The old `engine.py` contains a legacy implementation that has been marked as deprecated. All new development should use:

- **Entry point**: `engine_runner.py`
- **Enhanced engine**: `src/engine_enhanced.py`
- **Option chains**: `src/provider/option_chain_builder.py`
- **Provider**: `src/provider/kite.py`

## Configuration

Default settings can be adjusted:
- **Batch size**: Max 500 tokens (Kite limit)
- **Rate limiting**: 200ms between batches
- **Strike range**: ±20% around spot
- **Max strikes**: 100 strikes maximum
- **Cache TTL**: 5 minutes for expiry data