# modal_decomposition_and_flow_reconstruction_using_parametric_DMD
Parametric DMD for Flow Reconstruction: From Canonical Cylinder Validation to Transonic Buffet Modeling
## Overview

The accurate and efficient modeling of fluid flows across varying operating conditions is a central challenge in computational fluid dynamics (CFD), particularly in the context of reduced-order modeling. Parametric Dynamic Mode Decomposition (pDMD) has emerged as a powerful data-driven technique for extracting low-dimensional representations from high-fidelity simulations or experiments. It enables interpolation of system behavior across parameter spaces, such as Reynolds number, Mach number, or angle of attack.

However, like its foundational counterpart — vanilla DMD — pDMD remains sensitive to noise and nonlinear dynamics, which limits its predictive stability and robustness. This thesis aims to develop and validate an improved version of pDMD by integrating an optimized DMD variant developed at the Institute of Fluid Mechanics, TU Dresden.

The improved algorithm is evaluated on canonical benchmark cases, including laminar flow past a cylinder and transonic flow over an airfoil under stall conditions. The results demonstrate enhanced accuracy and long-term stability, as well as reduced sensitivity to input noise. The project further provides insights into the relationship between flow physics, parametric variation, and the structure of reduced-order models. While the improved pDMD generalizes well across similar flow regimes, its performance in extrapolative scenarios remains an open area for future research.
