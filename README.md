# Quantum Imaging with Undetected Light

Many molecules exhibit strong absorption features in the mid-infrared spectral range, while
cameras operating in this range are expensive, prone to noise, and offer lower resolution
than conventional Si visible-range cameras. Such challenges can be addressed by exploiting
quantum correlation between entangled photon pairs 
generated in processes such as Spontaneous Parametric Down Conversion (SPDC).
This allows an object to be probed with a photon of one wavelength,
while its correlated partner of another wavelength is detected.

Described approach, known as Quantum Imaging with Undetected Light (QIUL), decouples the
illumination and detection wavelengths, enabling imaging in spectral regions where suitable
detectors are unavailable or perform poorly at the sample probing wavelength, giving it
advantage over conventional imaging methods.

This repository contains classes for computational modeling of SPDC source
and the entire QIUL setup. It is based on the following work:

F. Riexinger,
_Simulation Methods for Spontaneous Parametric Down-Conversion and Quantum-Sensing Systems with Undetected Light_,
PhD Thesis (2023)

in which a detailed theoretical derivation can also be found.

## What's in this directory

- [src](src):
- [tests](test):

## SPDC Simulation

Spontaneous Parametric Down-Conversion is a second order nonlinear process in which
one _pump_ photon is converted into two photons of lower frequency, denoted as _signal_ and
_idler_. The significance of this process stems from its great flexibility when selecting
wavelengths, ability to work at room temperatures and under normal atmospheric
conditions, and possibility of generating various entangled states exploited in QIUL.

## QIUL Simulation

The experimental work focuses on the development and
optimization of an interferometric setup in Michelson
geometry. The system utilizes double-pass SPDC generation
in a periodically poled KTP (ppKTP) crystal. Due to the momentum
correlation between the idler and the signal from the first pass,
the signal photon has the information about the object,
although it never interacts with it. Signals from both passes
are then combined in the detector arm, and information about
object can be obtained from their interference.

## Prerequisits

All requirements are listed in [requirements.txt](requirements.txt).
Test scripts are in the form of Python Notebooks so make sure you can also run them.
