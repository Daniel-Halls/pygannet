# Gannet Mathematics: Peak Fitting and Optimization

This document outlines the mathematical models and optimization routines that Gannet uses to fit metabolite peaks in the frequency domain and extract quantitative parameters (like peak area and FWHM).

## 1. Peak Fitting Models

Gannet uses non-linear combinations of analytical functions to model the observed spectral data in the frequency domain. The choice of model depends on the target metabolite (e.g., Gaussian for GABA/Glx, Lorentzian for Ethanol). 

### 1.1 Single Gaussian Model (e.g., for GABA)
The simplest model Gannet uses is a single Gaussian curve overlaid on a linear baseline. The mathematical formulation is:

$$ F(x, f) = x_1 \exp\left( x_2 (f - x_3)^2 \right) + x_4 (f - x_3) + x_5 $$

Where:
*   $f$: Frequency vector (in ppm).
*   $x_1$: Amplitude (height) of the Gaussian peak.
*   $x_2$: A width parameter, strictly defined as $x_2 = -\frac{1}{2\sigma^2}$. Since $\sigma$ is the standard deviation, $x_2$ will be negative.
*   $x_3$: The center frequency of the peak ($f_0$).
*   $x_4$: The slope of the linear baseline.
*   $x_5$: The constant offset (y-intercept) of the baseline.

### 1.2 Multi-Gaussian Models (e.g., Double for Glx, Triple for GABA+Glx)
For targets with distinct multiplet structures or when co-fitting multiple metabolites (like GABA and Glx simultaneously), Gannet expands the Gaussian model additively. For example, the `GABAGlxModel` uses a **three-Gaussian** approach with an advanced baseline:

$$ F(x, f) = \underbrace{x_1 e^{x_2 (f - x_3)^2}}_{\text{Glx Peak 1}} + \underbrace{x_4 e^{x_5 (f - x_6)^2}}_{\text{Glx Peak 2}} + \underbrace{x_7 e^{x_8 (f - x_9)^2}}_{\text{GABA Peak}} + \underbrace{x_{10} (f - x_3) + x_{11} \sin(\alpha f) + x_{12} \cos(\alpha f)}_{\text{Baseline Model}} $$

*Note: $\alpha = \frac{\pi}{1.31 \times 4}$ is a constant scaling factor used for the sinusoidal baseline in the GABA+Glx model to account for broader macro-molecule or co-edited background distortions.*

### 1.3 Lorentzian Model (e.g., for Ethanol)
Some metabolites are better described by Lorentzian lineshapes, which have wider tails than Gaussians. Gannet models the Lorentzian incorporating both **absorption** and **dispersion** components to account for slight phase errors:

$$ \text{Absorption}(f) = \frac{1}{2\pi} \frac{x_1 \cdot x_2}{(f - x_3)^2 + x_2^2} $$
$$ \text{Dispersion}(f) = \frac{1}{2\pi} \frac{x_1 \cdot (f - x_3)}{(f - x_3)^2 + x_2^2} $$

The full model is then:

$$ F(x, f) = \cos(x_4) \cdot \text{Absorption}(f) + \sin(x_4) \cdot \text{Dispersion}(f) + x_5 + x_6 (f - x_3) $$

Where:
*   $x_1$: Area of the peak.
*   $x_2$: Half-width at half-maximum (HWHM).
*   $x_3$: Center frequency ($f_0$).
*   $x_4$: Phase angle (mixing absorption and dispersion).
*   $x_5$, $x_6$: Baseline constants (offset and slope).

---

## 2. The Optimization Process

Once the model $F(x, f)$ is defined, Gannet's goal is to find the optimal parameter set $x$ that best describes the observed data $y$. Gannet achieves this through a two-step optimization process using MATLAB's built-in statistical functions.

### 2.1 The Objective Function (Weighted Sum of Squared Residuals)
The mathematical goal is to minimize the weighted sum of squared differences (residuals) between the model and the actual spectrum. The loss function $L(x)$ is defined as:

$$ L(x) = \sum_{i} w_i \left[ y(f_i) - F(x, f_i) \right]^2 $$

Where:
*   $y(f_i)$: The actual signal amplitude (from the DIFF spectrum) at frequency bin $f_i$.
*   $F(x, f_i)$: The predicted signal amplitude from the mathematical model.
*   $w_i$: A weight assigned to the $i$-th frequency bin.

**Why weights?** Gannet uses weights to intentionally "ignore" or down-weight certain frequency ranges where known artifacts occur. For example, when fitting GABA around 3.02 ppm, there is a large, often distorted subtraction artifact from Choline (Cho) near 3.2 ppm. Gannet sets $w_i = 0.001$ for the Cho range and $w_i = 1$ everywhere else, preventing the optimization algorithm from distorting the GABA fit to try and accommodate the Cho artifact.

### 2.2 Step 1: Bounded Optimization (`lsqcurvefit`)
Gannet first uses `lsqcurvefit` to find an initial rough fit. 
*   **Algorithm:** Typically the Trust-Region-Reflective least-squares algorithm.
*   **Role:** It enforces strict lower (`lb`) and upper (`ub`) boundaries on the parameters. For instance, the GABA amplitude cannot be negative, and the center frequency is bounded to stay very close to 3.02 ppm. 
*   **Math Concept:** It solves $\min_{x} || F(x, f) - y ||_2^2$ subject to $lb \le x \le ub$. This ensures the model doesn't drift into mathematically valid but physically impossible parameter spaces.

### 2.3 Step 2: Weighted Unconstrained Optimization (`nlinfit`)
After finding a bounded estimate, Gannet passes those parameters as the initial guess into `nlinfit`.
*   **Algorithm:** Levenberg-Marquardt algorithm.
*   **Role:** This step refines the fit and incorporates the weighting vector $w$. The Levenberg-Marquardt algorithm interpolates between the Gauss-Newton algorithm (fast convergence near the minimum) and the method of gradient descent (robust when far from the minimum). 
*   **Output:** It outputs the final optimized parameters $x_{opt}$ and the residual vector.

---

## 3. Deriving Quantitative Metrics

After the optimal parameters $x_{opt}$ are found, Gannet derives the clinically relevant metrics: Peak Area and Full-Width Half-Maximum (FWHM).

**For a Gaussian Model:**
Recall that $x_1$ is the amplitude and $x_2 = -\frac{1}{2\sigma^2}$. 
*   **Standard Deviation ($\sigma$):** $\sigma = \sqrt{\frac{1}{2|x_2|}}$
*   **FWHM:** The FWHM of a Gaussian is $2\sqrt{2 \ln 2} \sigma$, but Gannet converts this to Hz using the scanner's Larmor frequency ($L_f$ in MHz). Gannet calculates it as: 
    $$ \text{FWHM (Hz)} = \left(2 \cdot L_f\right) \cdot \sigma $$
*   **Peak Area:** The integral of a Gaussian function $A e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ is $A\sigma\sqrt{2\pi}$. Gannet computes this as:
    $$ \text{Area} = x_1 \sqrt{\pi} \cdot \frac{1}{\sqrt{|x_2|}} $$

These area values (e.g., Area of GABA vs. Area of unsuppressed Water) are then passed to `GannetQuantify.m` to calculate absolute concentrations.
