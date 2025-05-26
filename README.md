# DWARF
Dynamic Wake Accretion in Relativistic Fluids Theory
This is a repository to keep my work safe, as I try to push my theory to the limits.

Current Tasks:

Currently running a simulation that is:
 - Building a system where gravity is emergent, not imposed.
 - Seeing coherent, stable, repeatable entrainment, the very mechanism that makes real-world orbits possible, without invoking Newton or Einstein. - currently running and working as predicted.
 - Optimizing with GPU (CUDA) - DONE
 - Results are incredible so far, Next step a full emergent cosmogenesis engine?
 - Paper on results to begin shortly

-------------------------------------------------------------------
- Getting cupy running on my local machine to utilize GPU for faster sims - Done
- Building a fluid simulator - (starting to look like fluid engine)
  - Fluid Simulator upgraded to a cupy model with all math findings from INFLOW, Solar, and Tidal sims.
  - 2000 step Vortex sim Ran with DWARF as the model
   - DWARF’s ring radius aligns tightly with the Lamb-Oseen model.
   - DWARF Derives Viscous Behavior Without Traditional Navier-Stokes Terms
   - Wake memory introduces a novel, tunable mechanism for field-aware dissipation in fluids — this has no analogue in standard physics.
   - Stability can emerge from saturation
   - Topological fluid behavior can emerge from pure flow entrainment
   - DWARF’s nonlinearity introduces natural gradient saturation, allowing stable computation where classic solvers struggle.
 
  - 
- Building an INFLOW simulator - Started, tested and results are awesome (will upload soon)
- Overlaying findings between them
- Building an Orbital Mechanics Simulator once we have mechanics from Fluid and INFLOW - almost done, but not with INFLOW yet
- Tidal Simulator updated with math findings, almost producing perfect real  world results using DWARF as primary principle
---Side thoughts, considering merging DWARF_V1.2 and v1.3 and making it a new paper, or just leaving them as Git Documents, and moving into my new findings Via all the simulations I have been running---
