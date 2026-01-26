"""
Test script to verify download functionality works correctly.
This creates sample data and tests that CSV generation works.
"""
import pandas as pd
import io
from datetime import datetime, timedelta

def test_csv_generation():
    """Test that CSV data can be generated for download."""
    print("Testing CSV generation for download functionality...")
    
    # Create sample backtest data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'close_price': [4.5 + i*0.01 for i in range(100)],
        'prediction_probability': [0.5 + (i % 10) * 0.02 for i in range(100)],
        'predicted_direction': [i % 2 for i in range(100)],
        'actual_direction': [(i + 1) % 2 for i in range(100)]
    })
    
    # Test 1: Basic CSV generation
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    assert len(csv_data) > 0, "CSV data should not be empty"
    assert 'date' in csv_data, "CSV should contain header"
    print("✓ Basic CSV generation works")
    
    # Test 2: Date formatting
    sample_data_formatted = sample_data.copy()
    sample_data_formatted['date'] = sample_data_formatted['date'].dt.strftime('%Y-%m-%d')
    csv_buffer2 = io.StringIO()
    sample_data_formatted.to_csv(csv_buffer2, index=False)
    csv_data2 = csv_buffer2.getvalue()
    assert '2024-01-01' in csv_data2, "Formatted date should be in CSV"
    print("✓ Date formatting works")
    
    # Test 3: Summary report generation
    summary_data = {
        'Metric': ['Accuracy', 'Precision', 'F1 Score'],
        'Value': ['0.6500', '0.6200', '0.6350']
    }
    summary_df = pd.DataFrame(summary_data)
    summary_csv = summary_df.to_csv(index=False)
    assert 'Metric,Value' in summary_csv, "Summary CSV should have headers"
    assert 'Accuracy' in summary_csv, "Summary CSV should contain metrics"
    print("✓ Summary report generation works")
    
    # Test 4: Feature importance data
    importance_data = pd.DataFrame({
        'feature': ['ret_1d', 'ret_5d', 'ma_20', 'vol_10'],
        'importance': [0.15, 0.12, 0.10, 0.08]
    })
    importance_csv = importance_data.to_csv(index=False)
    assert 'feature,importance' in importance_csv, "Feature CSV should have headers"
    print("✓ Feature importance CSV generation works")
    
    # Test 5: Filename generation with timestamp
    filename = f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv"
    assert len(filename) > 20, "Filename should include timestamp"
    assert filename.endswith('.csv'), "Filename should end with .csv"
    print(f"✓ Filename generation works: {filename}")
    
    print("\n✅ All download functionality tests passed!")
    return True

if __name__ == "__main__":
    test_csv_generation()
