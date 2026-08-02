# Ecnetica-Demo
Ecnetica BKT concept graph demo (floating/vibrating version)
==============================================================

Double-click Ecnetica_demo.html to open it in any browser - no installs
needed.

Nodes float gently and drift on their own. Click Correct or Wrong on
any concept and the graph reacts: that concept's mastery updates via
Bayesian Knowledge Tracing, then the update propagates outward to
related concepts, causing them to vibrate proportionally to how
strongly and how closely they're connected.

Controls
--------
- Mastery threshold slider: concepts turn green once their mastery
  estimate crosses this value.
- Base beta (attenuation) slider: controls how quickly an update's
  effect fades out as it spreads further across the graph.
- Correct / Wrong buttons per concept.
- Reset All Concept Masteries: returns every concept to its starting
  value.

This is a toy/demo simulation for working out the mechanics, not the
production Ecnetica model. Edge weighting between concepts is still an
open design question and isn't wired in yet - propagation strength
here depends only on graph distance and the base beta.

Credits
-------
Developed as part of the TCD Problem Solving Association.
Michael Mitchell acted as supervisor for the project.
