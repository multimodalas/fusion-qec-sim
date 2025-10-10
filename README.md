# fusion-qec-sim

A modular quantum error correction toolkit designed for global and LMIC (Low- and Middle-Income Country) labs. This toolkit provides hybrid quantum error correction simulations combining Steane CSS codes and Surface codes with Minimum Weight Perfect Matching (MWPM) decoding.

## Overview

`fusion-qec-sim` is a comprehensive simulation and diagnostics bundle for quantum error correction research and education. It integrates two powerful QEC approaches:

- **Steane CSS Codes**: Calderbank-Shor-Steane codes for quantum error correction
- **Surface Codes with MWPM**: Surface code implementation with Minimum Weight Perfect Matching decoder for syndrome analysis

The toolkit includes syndrome fusion analytics, automated reproducibility tools, and extensive diagnostics for hybrid quantum error correction schemes.

## Features

- **Modular Architecture**: Flexible design allowing easy integration and customization of QEC protocols
- **Hybrid QEC Simulations**: Combined Steane and Surface code error correction with fusion diagnostics
- **Fidelity Analysis**: Comprehensive error rate and fidelity tracking across different noise models
- **Data Export**: CSV file exports for all simulation results and metrics
- **Visualization Tools**: Automated generation of error rate charts and performance graphs
- **MWPM Decoding**: Efficient Minimum Weight Perfect Matching syndrome decoder for surface codes
- **Reproducibility**: Automated tools for reproducible experiments and benchmarking

## Code Structure

The toolkit provides modular components for:
- Quantum circuit simulation with noise models
- Steane code encoding and syndrome extraction
- Surface code lattice generation and syndrome measurement
- MWPM graph construction and optimal matching
- Error correction and logical error tracking
- Statistical analysis and visualization

## Usage

```python
# Example usage for Steane code simulation
from fusion_qec_sim import SteaneCode, NoiseModel

# Initialize Steane code
steane = SteaneCode()
noise = NoiseModel(physical_error_rate=0.001)

# Run simulation
results = steane.simulate(num_shots=10000, noise_model=noise)

# Export results to CSV
results.export_csv("steane_results.csv")

# Generate fidelity charts
results.plot_error_rates("steane_fidelity.png")
```

```python
# Example usage for Surface code with MWPM
from fusion_qec_sim import SurfaceCode, MWPMDecoder

# Initialize surface code
surface = SurfaceCode(distance=5)
decoder = MWPMDecoder()

# Run simulation with MWPM decoding
results = surface.simulate(
    num_shots=10000,
    physical_error_rate=0.001,
    decoder=decoder
)

# Export and visualize
results.export_csv("surface_results.csv")
results.plot_logical_error_rate("surface_performance.png")
```

## CSV Exports and Charts

All simulation results can be exported to CSV format for further analysis:
- Error rates vs physical error rates
- Logical error rates for different code distances
- Syndrome statistics and decoder performance metrics
- Fidelity measurements across different configurations

Charts and visualizations are automatically generated in PNG/PDF formats showing:
- Error correction thresholds
- Scaling behavior with code distance
- Comparative performance of Steane vs Surface codes
- Fusion diagnostic metrics

## Ethics and Responsible Use

This toolkit is designed to promote equitable access to quantum error correction research:

- **Educational Access**: Optimized for use in resource-constrained environments
- **Open Science**: All code and methodologies are transparent and reproducible
- **Global Collaboration**: Designed to support researchers in LMIC and global labs
- **Responsible Research**: Encourages ethical practices in quantum computing research

Users are encouraged to:
- Share improvements and results with the community
- Cite the toolkit in academic publications
- Contribute to making quantum computing education more accessible
- Follow ethical guidelines in computational research

## Installation

```bash
pip install fusion-qec-sim
```

Or install from source:
```bash
git clone https://github.com/multimodalas/fusion-qec-sim.git
cd fusion-qec-sim
pip install -e .
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for visualization)
- NetworkX (for MWPM graph algorithms)

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest new features. We particularly encourage contributions that:
- Improve accessibility for LMIC labs
- Add new QEC protocols or decoders
- Enhance documentation and tutorials
- Optimize performance for limited computational resources

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Project Repository**: [https://github.com/multimodalas/fusion-qec-sim](https://github.com/multimodalas/fusion-qec-sim)
- **Issues and Support**: [https://github.com/multimodalas/fusion-qec-sim/issues](https://github.com/multimodalas/fusion-qec-sim/issues)
- **Maintainer**: multimodalas

For questions, collaborations, or support requests, please open an issue on GitHub or contact the maintainers through the repository.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{fusion_qec_sim,
  title = {fusion-qec-sim: A Modular Quantum Error Correction Toolkit},
  author = {multimodalas},
  year = {2025},
  url = {https://github.com/multimodalas/fusion-qec-sim},
  license = {MIT}
}
```

## Acknowledgments

This toolkit is developed to support quantum error correction research in global and LMIC laboratory settings, promoting equitable access to advanced quantum computing tools and education.
