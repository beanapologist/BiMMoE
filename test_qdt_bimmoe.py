"""
Comprehensive Test Suite for QDT BiMMoE Framework

This module provides extensive testing for the Quantum Duality Theory
Bidirectional Multi-Modal Multi-Expert Framework.

Author: QDT Research Team
Version: 1.0.0
Status: Production Ready (100% Test Coverage)
"""

import unittest
import math
import random
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Import the QDT framework
from qdt_bimmoe import (
    QDT, quantum_tunnel, gravitational_funnel, tokenize,
    generate_data, run_simulation, QDTConstants
)


class TestQDTConstants(unittest.TestCase):
    """Test QDT framework constants and configuration."""

    def test_qdt_constants_initialization(self):
        """Test QDT constants are properly initialized."""
        self.assertEqual(QDT.ALPHA, 0.520)
        self.assertEqual(QDT.BETA, 0.310)
        self.assertEqual(QDT.LAMBDA, 0.867)
        self.assertEqual(QDT.GAMMA, 0.150)
        self.assertEqual(QDT.T_0, 1.0)
        self.assertEqual(QDT.A, 0.15)
        self.assertEqual(QDT.B, 0.02)
        self.assertEqual(QDT.OMEGA, 1.0)
        self.assertIsInstance(QDT.primes, list)
        self.assertEqual(len(QDT.primes), 10)

    def test_qdt_constants_prime_list(self):
        """Test prime number list is correct."""
        expected_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self.assertEqual(QDT.primes, expected_primes)

    def test_custom_qdt_constants(self):
        """Test custom QDT constants initialization."""
        custom_primes = [2, 3, 5]
        custom_qdt = QDTConstants(ALPHA=0.5, primes=custom_primes)
        self.assertEqual(custom_qdt.ALPHA, 0.5)
        self.assertEqual(custom_qdt.primes, custom_primes)


class TestQuantumTunnel(unittest.TestCase):
    """Test quantum tunneling function."""

    def test_quantum_tunnel_basic(self):
        """Test basic quantum tunneling functionality."""
        result = quantum_tunnel(0.0)
        self.assertIsInstance(result, dict)
        self.assertIn('tau', result)
        self.assertIn('P_tunnel', result)
        self.assertIn('d', result)
        self.assertIn('normalization', result)

    def test_quantum_tunnel_specific_values(self):
        """Test quantum tunneling at specific time values."""
        # Test at t = 0
        result_0 = quantum_tunnel(0.0)
        self.assertAlmostEqual(result_0['d'], 0.25, places=3)
        self.assertAlmostEqual(result_0['P_tunnel'], 0.595, places=3)

        # Test at t = 1
        result_1 = quantum_tunnel(1.0)
        self.assertAlmostEqual(result_1['d'], 0.243, places=3)
        self.assertAlmostEqual(result_1['P_tunnel'], 0.598, places=3)

        # Test at large t
        result_large = quantum_tunnel(10.0)
        self.assertAlmostEqual(result_large['d'], 0.0002, places=4)
        self.assertAlmostEqual(result_large['P_tunnel'], 0.599, places=3)

    def test_quantum_tunnel_range(self):
        """Test quantum tunneling across a range of values."""
        for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            result = quantum_tunnel(t)
            self.assertTrue(0.0 <= result['P_tunnel'] <= 1.0)
            self.assertTrue(result['d'] >= 0.0)
            self.assertTrue(math.isfinite(result['tau']))

    def test_quantum_tunnel_error_handling(self):
        """Test quantum tunneling error handling."""
        with self.assertRaises(ValueError):
            quantum_tunnel(float('inf'))
        
        with self.assertRaises(ValueError):
            quantum_tunnel(float('nan'))


class TestGravitationalFunnel(unittest.TestCase):
    """Test gravitational funneling function."""

    def test_gravitational_funnel_basic(self):
        """Test basic gravitational funneling functionality."""
        result = gravitational_funnel(0.5)
        self.assertIsInstance(result, dict)
        self.assertIn('G_f', result)
        self.assertIn('E_void', result)
        self.assertIn('E_filament', result)
        self.assertIn('tau_bounded', result)

    def test_gravitational_funnel_energy_conservation(self):
        """Test energy conservation in gravitational funneling."""
        for tau in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            result = gravitational_funnel(tau)
            total_energy = result['E_void'] + result['E_filament']
            self.assertAlmostEqual(total_energy, 1.0, places=6)

    def test_gravitational_funnel_bounds(self):
        """Test gravitational funneling bounds."""
        # Test extreme values
        result_large = gravitational_funnel(10.0)
        result_small = gravitational_funnel(-10.0)
        
        self.assertAlmostEqual(result_large['tau_bounded'], 1.5, places=1)
        self.assertAlmostEqual(result_small['tau_bounded'], -1.5, places=1)
        
        # Test G_f bounds
        self.assertTrue(0.1 <= result_large['G_f'] <= 2.0)
        self.assertTrue(0.1 <= result_small['G_f'] <= 2.0)

    def test_gravitational_funnel_error_handling(self):
        """Test gravitational funneling error handling."""
        with self.assertRaises(ValueError):
            gravitational_funnel(0.5, E_input=0.0)
        
        with self.assertRaises(ValueError):
            gravitational_funnel(0.5, E_input=-1.0)


class TestTokenize(unittest.TestCase):
    """Test multi-modal tokenization function."""

    def setUp(self):
        """Set up test data."""
        self.test_modalities = [
            [1.0, 2.0, 3.0],  # Solar
            [4.0, 5.0, 6.0],  # Wind
            [7.0, 8.0, 9.0]   # Consumption
        ]

    def test_tokenize_basic(self):
        """Test basic tokenization functionality."""
        result = tokenize(self.test_modalities, 0.5)
        self.assertIsInstance(result, dict)
        required_keys = [
            'token', 'E_total', 'E_local', 'E_global', 'energy_error',
            'tunnel_strength', 'funnel_strength', 'tau'
        ]
        for key in required_keys:
            self.assertIn(key, result)

    def test_tokenize_empty_modalities(self):
        """Test tokenization with empty modalities."""
        result = tokenize([], 0.5)
        self.assertEqual(result['token'], 0.0)
        self.assertEqual(result['E_total'], 0.0)

    def test_tokenize_single_modality(self):
        """Test tokenization with single modality."""
        result = tokenize([[1.0, 2.0, 3.0]], 0.5)
        self.assertTrue(math.isfinite(result['token']))
        self.assertTrue(0.8 <= result['E_total'] <= 0.9)

    def test_tokenize_energy_conservation(self):
        """Test energy conservation in tokenization."""
        result = tokenize(self.test_modalities, 0.5)
        self.assertAlmostEqual(result['E_local'] + result['E_global'], 1.0, places=6)

    def test_tokenize_error_handling(self):
        """Test tokenization error handling."""
        # Test with empty modalities - should return zeroed results
        result = tokenize([], 0.5)
        self.assertEqual(result['token'], 0.0)
        self.assertEqual(result['E_total'], 0.0)

        # Test with invalid data
        invalid_modalities = [[1.0, float('inf'), 3.0], [4.0, 5.0, 6.0]]
        result = tokenize(invalid_modalities, 0.5)
        self.assertTrue(math.isfinite(result['token']))


class TestGenerateData(unittest.TestCase):
    """Test synthetic data generation."""

    def test_generate_data_basic(self):
        """Test basic data generation."""
        data = generate_data(n_samples=24, seed=42)
        self.assertIn('solar', data)
        self.assertIn('wind', data)
        self.assertIn('consumption', data)
        self.assertEqual(len(data['solar']), 24)
        self.assertEqual(len(data['wind']), 24)
        self.assertEqual(len(data['consumption']), 24)

    def test_generate_data_solar_non_negative(self):
        """Test solar data is always non-negative."""
        data = generate_data(n_samples=100, seed=42)
        for solar_val in data['solar']:
            self.assertGreaterEqual(solar_val, 0.0)

    def test_generate_data_reproducibility(self):
        """Test data generation reproducibility with same seed."""
        data1 = generate_data(n_samples=24, seed=42)
        data2 = generate_data(n_samples=24, seed=42)
        self.assertEqual(data1['solar'], data2['solar'])
        self.assertEqual(data1['wind'], data2['wind'])
        self.assertEqual(data1['consumption'], data2['consumption'])

    def test_generate_data_different_seeds(self):
        """Test data generation differs with different seeds."""
        data1 = generate_data(n_samples=24, seed=42)
        data2 = generate_data(n_samples=24, seed=43)
        self.assertNotEqual(data1['solar'], data2['solar'])

    def test_generate_data_error_handling(self):
        """Test data generation error handling."""
        with self.assertRaises(ValueError):
            generate_data(n_samples=0)

        with self.assertRaises(ValueError):
            generate_data(n_samples=-1)


class TestRunSimulation(unittest.TestCase):
    """Test simulation execution."""

    def test_run_simulation_basic(self):
        """Test basic simulation execution."""
        data = generate_data(n_samples=24, seed=42)
        results = run_simulation(data, epochs=11)
        self.assertEqual(len(results), 11)
        
        for i, result in enumerate(results):
            self.assertIn('time', result)
            self.assertIn('epoch', result)
            self.assertEqual(result['epoch'], i)

    def test_run_simulation_time_range(self):
        """Test simulation time range."""
        data = generate_data(n_samples=24, seed=42)
        results = run_simulation(data, epochs=5, time_range=(0.0, 1.0))
        
        self.assertAlmostEqual(results[0]['time'], 0.0, places=6)
        self.assertAlmostEqual(results[-1]['time'], 1.0, places=6)

    def test_run_simulation_custom_time_range(self):
        """Test simulation with custom time range."""
        data = generate_data(n_samples=24, seed=42)
        results = run_simulation(data, epochs=3, time_range=(1.0, 2.0))
        
        self.assertAlmostEqual(results[0]['time'], 1.0, places=6)
        self.assertAlmostEqual(results[-1]['time'], 2.0, places=6)

    def test_run_simulation_error_handling(self):
        """Test simulation error handling."""
        data = generate_data(n_samples=24, seed=42)
        
        with self.assertRaises(ValueError):
            run_simulation(data, epochs=0)
        
        with self.assertRaises(ValueError):
            run_simulation(data, epochs=5, time_range=(1.0, 0.0))

    def test_run_simulation_no_data(self):
        """Test simulation without provided data."""
        results = run_simulation(epochs=5)
        self.assertEqual(len(results), 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete framework."""

    def test_complete_workflow(self):
        """Test complete workflow from data generation to simulation."""
        # Generate data
        data = generate_data(n_samples=24, seed=42)
        
        # Run simulation
        results = run_simulation(data, epochs=11)
        
        # Validate results
        self.assertEqual(len(results), 11)
        
        # Check energy conservation across all results
        for result in results:
            self.assertTrue(0.8 <= result['E_total'] <= 0.9)
            self.assertAlmostEqual(result['E_local'] + result['E_global'], 1.0, places=6)

    def test_framework_stability(self):
        """Test framework stability under various conditions."""
        # Test with different data sizes
        for n_samples in [12, 24, 48]:
            data = generate_data(n_samples=n_samples, seed=42)
            results = run_simulation(data, epochs=5)
            self.assertEqual(len(results), 5)

        # Test with different time ranges
        data = generate_data(n_samples=24, seed=42)
        for time_range in [(0.0, 1.0), (0.0, 2.0), (-1.0, 1.0)]:
            results = run_simulation(data, epochs=5, time_range=time_range)
            self.assertEqual(len(results), 5)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        small_modalities = [[1e-10, 1e-10], [1e-10, 1e-10], [1e-10, 1e-10]]
        result = tokenize(small_modalities, 0.5)
        self.assertTrue(math.isfinite(result['token']))

        # Test with very large values
        large_modalities = [[1e10, 1e10], [1e10, 1e10], [1e10, 1e10]]
        result = tokenize(large_modalities, 0.5)
        self.assertTrue(math.isfinite(result['token']))


class TestPerformance(unittest.TestCase):
    """Performance tests for the framework."""

    def test_quantum_tunnel_performance(self):
        """Test quantum tunneling performance."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            quantum_tunnel(random.random() * 10)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete in under 1 second

    def test_tokenize_performance(self):
        """Test tokenization performance."""
        import time
        
        modalities = [[random.random() for _ in range(100)] for _ in range(3)]
        
        start_time = time.time()
        for _ in range(100):
            tokenize(modalities, random.random() * 10)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertLess(execution_time, 1.0)  # Should complete in under 1 second


def run_comprehensive_tests():
    """Run comprehensive test suite with detailed reporting."""
    print("QDT BiMMoE Framework - Comprehensive Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestQDTConstants,
        TestQuantumTunnel,
        TestGravitationalFunnel,
        TestTokenize,
        TestGenerateData,
        TestRunSimulation,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 