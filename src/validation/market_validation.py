#!/usr/bin/env python3
"""
Market Validation Framework

This module provides validation capabilities for comparing calculated values
against market references like Sensibull, but does NOT use them as ground truth.
The calculations are based purely on mathematical formulas and real market data.
"""

from __future__ import annotations
import datetime as dt
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class MarketValidationFramework:
    """
    Framework for validating calculations against market references.
    
    NOTE: Reference values (like Sensibull) are used only for comparison
    and validation purposes. They are NOT used as ground truth or for
    parameter tuning.
    """
    
    def __init__(self):
        self.reference_sources = ["sensibull", "manual_calculation"]
        
    def compare_with_reference(
        self,
        calculated_values: Dict[str, Any],
        reference_values: Dict[str, Any],
        reference_source: str = "sensibull",
        tolerance: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Compare calculated values with reference values for validation.
        
        This is used to validate our mathematical implementations, NOT to
        tune parameters or normalize results.
        """
        if tolerance is None:
            tolerance = {
                "max_pain": 100.0,   # Points difference allowed
                "atm": 100.0,        # Points difference allowed  
                "pcr": 0.05,         # Absolute difference allowed
                "atm_iv": 1.0        # Percentage points difference allowed
            }
        
        comparison = {
            "reference_source": reference_source,
            "timestamp": dt.datetime.now(),
            "symbol": calculated_values.get("symbol", "unknown"),
            "comparison_results": {},
            "overall_assessment": "pending"
        }
        
        metrics_to_compare = ["max_pain", "atm", "pcr", "atm_iv"]
        
        for metric in metrics_to_compare:
            calc_value = calculated_values.get(metric)
            ref_value = reference_values.get(metric)
            
            result = {
                "calculated": calc_value,
                "reference": ref_value,
                "difference": None,
                "within_tolerance": False,
                "status": "not_compared"
            }
            
            if calc_value is not None and ref_value is not None:
                if metric == "pcr":
                    # PCR is ratio, use absolute difference
                    difference = abs(calc_value - ref_value)
                    result["difference"] = difference
                    result["within_tolerance"] = difference <= tolerance[metric]
                elif metric == "atm_iv":
                    # IV is percentage, compare as percentage points
                    difference = abs(calc_value - ref_value)
                    result["difference"] = difference
                    result["within_tolerance"] = difference <= tolerance[metric]
                else:
                    # Max Pain and ATM are prices in points
                    difference = abs(calc_value - ref_value)
                    result["difference"] = difference
                    result["within_tolerance"] = difference <= tolerance[metric]
                
                result["status"] = "within_tolerance" if result["within_tolerance"] else "outside_tolerance"
            
            elif calc_value is not None:
                result["status"] = "no_reference"
            elif ref_value is not None:
                result["status"] = "calculation_failed"
            else:
                result["status"] = "both_missing"
            
            comparison["comparison_results"][metric] = result
        
        # Overall assessment
        successful_comparisons = [
            r for r in comparison["comparison_results"].values() 
            if r["status"] in ["within_tolerance", "outside_tolerance"]
        ]
        
        if not successful_comparisons:
            comparison["overall_assessment"] = "no_valid_comparisons"
        else:
            within_tolerance_count = sum(
                1 for r in successful_comparisons if r["within_tolerance"]
            )
            tolerance_rate = within_tolerance_count / len(successful_comparisons)
            
            if tolerance_rate >= 0.75:
                comparison["overall_assessment"] = "good_agreement"
            elif tolerance_rate >= 0.5:
                comparison["overall_assessment"] = "moderate_agreement"
            else:
                comparison["overall_assessment"] = "poor_agreement"
        
        return comparison
    
    def validate_calculation_quality(
        self, calculated_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the quality of calculations based on mathematical consistency.
        
        This checks internal consistency and mathematical validity, not
        comparison with external references.
        """
        validation = {
            "timestamp": dt.datetime.now(),
            "symbol": calculated_values.get("symbol", "unknown"),
            "quality_checks": {},
            "overall_quality": "pending"
        }
        
        # Check Max Pain reasonableness
        max_pain = calculated_values.get("max_pain")
        spot = calculated_values.get("spot")
        if max_pain is not None and spot is not None:
            mp_spot_ratio = abs(max_pain - spot) / spot
            validation["quality_checks"]["max_pain_reasonableness"] = {
                "check": "max_pain_vs_spot",
                "ratio": mp_spot_ratio,
                "reasonable": mp_spot_ratio < 0.1,  # Max Pain within 10% of spot
                "comment": "Max Pain should be relatively close to spot price"
            }
        
        # Check ATM vs Spot relationship
        atm = calculated_values.get("atm")
        if atm is not None and spot is not None:
            atm_spot_ratio = abs(atm - spot) / spot
            validation["quality_checks"]["atm_reasonableness"] = {
                "check": "atm_vs_spot",
                "ratio": atm_spot_ratio,
                "reasonable": atm_spot_ratio < 0.05,  # ATM within 5% of spot
                "comment": "ATM should be close to spot/forward price"
            }
        
        # Check PCR reasonableness
        pcr = calculated_values.get("pcr")
        if pcr is not None:
            validation["quality_checks"]["pcr_reasonableness"] = {
                "check": "pcr_range",
                "value": pcr,
                "reasonable": 0.1 <= pcr <= 5.0,  # PCR in reasonable range
                "comment": "PCR should be between 0.1 and 5.0 for normal markets"
            }
        
        # Check ATM IV reasonableness
        atm_iv = calculated_values.get("atm_iv")
        if atm_iv is not None:
            iv_percentage = atm_iv if atm_iv > 1 else atm_iv * 100
            validation["quality_checks"]["iv_reasonableness"] = {
                "check": "atm_iv_range",
                "value": iv_percentage,
                "reasonable": 5.0 <= iv_percentage <= 50.0,  # IV between 5% and 50%
                "comment": "ATM IV should be between 5% and 50% for normal markets"
            }
        
        # Overall quality assessment
        quality_checks = validation["quality_checks"]
        if not quality_checks:
            validation["overall_quality"] = "no_checks_performed"
        else:
            reasonable_count = sum(
                1 for check in quality_checks.values() if check.get("reasonable", False)
            )
            quality_rate = reasonable_count / len(quality_checks)
            
            if quality_rate >= 0.8:
                validation["overall_quality"] = "high_quality"
            elif quality_rate >= 0.6:
                validation["overall_quality"] = "moderate_quality"
            else:
                validation["overall_quality"] = "low_quality"
        
        return validation
    
    def generate_validation_report(
        self,
        calculated_values: Dict[str, Any],
        reference_values: Optional[Dict[str, Any]] = None,
        reference_source: str = "sensibull"
    ) -> str:
        """Generate a human-readable validation report."""
        report_lines = []
        symbol = calculated_values.get("symbol", "unknown")
        
        report_lines.append(f"VALIDATION REPORT - {symbol}")
        report_lines.append("=" * 50)
        
        # Quality validation
        quality = self.validate_calculation_quality(calculated_values)
        report_lines.append(f"\n1. MATHEMATICAL QUALITY: {quality['overall_quality'].upper()}")
        
        for check_name, check_result in quality["quality_checks"].items():
            status = "✓" if check_result["reasonable"] else "⚠"
            report_lines.append(f"   {status} {check_result['comment']}")
        
        # Reference comparison (if provided)
        if reference_values:
            comparison = self.compare_with_reference(
                calculated_values, reference_values, reference_source
            )
            report_lines.append(f"\n2. REFERENCE COMPARISON ({reference_source.upper()}): {comparison['overall_assessment'].upper()}")
            
            for metric, result in comparison["comparison_results"].items():
                if result["status"] in ["within_tolerance", "outside_tolerance"]:
                    status = "✓" if result["within_tolerance"] else "⚠"
                    calc_val = result["calculated"]
                    ref_val = result["reference"]
                    diff = result["difference"]
                    report_lines.append(f"   {status} {metric.upper()}: Calc={calc_val}, Ref={ref_val}, Diff={diff:.3f}")
        
        # Calculated values summary
        report_lines.append(f"\n3. CALCULATED VALUES:")
        report_lines.append(f"   Spot: {calculated_values.get('spot', 'N/A')}")
        report_lines.append(f"   Max Pain: {calculated_values.get('max_pain', 'N/A')}")
        report_lines.append(f"   ATM: {calculated_values.get('atm', 'N/A')}")
        report_lines.append(f"   PCR: {calculated_values.get('pcr', 'N/A')}")
        report_lines.append(f"   ATM IV: {calculated_values.get('atm_iv', 'N/A')}")
        
        report_lines.append("\nNOTE: All calculations based on standard mathematical formulas")
        report_lines.append("using real-time market data from Kite Connect.")
        
        return "\n".join(report_lines)